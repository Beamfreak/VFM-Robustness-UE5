"""
Main evaluation entry point for Vision Model Evaluation Framework.

Usage:
    python -m scripts.evaluate                    # Evaluate all models
    python -m scripts.evaluate --models clip_b resnet50  # Specific models
    python -m scripts.evaluate --models clip_b --skip-inference  # Metrics only
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from .config import (
    Config, MODELS, BASELINE_MODEL,
    get_default_config, DATASETS
)
from .utils import (
    set_seed, set_deterministic, print_header, 
    format_time, create_run_manifest, save_json, ensure_dir
)
from .models import ModelLoader, print_available_models
from .data_loader import (
    load_preprocessed_metadata, create_dataloader,
    get_dataset_statistics,
)
from .inference import run_model_inference, save_predictions, load_predictions
from .metrics import compute_metrics_from_predictions
from .analysis import run_stratified_analysis
from .reporting import (
    generate_model_report, generate_comparative_report
)


VARIANT_SEPARATOR = "__"


def build_model_variant_key(model_key: str, eval_mode: str) -> str:
    """Build a stable model-variant key."""
    return f"{model_key}{VARIANT_SEPARATOR}{eval_mode}"


def parse_model_variant_key(model_variant_key: str) -> tuple[str, str]:
    """Parse a model-variant key into (base_model_key, eval_mode)."""
    if VARIANT_SEPARATOR in model_variant_key:
        base_key, eval_mode = model_variant_key.split(VARIANT_SEPARATOR, 1)
        return base_key, eval_mode

    # Backward compatibility: treat bare keys according to configured default mode.
    default_mode = MODELS[model_variant_key].eval_mode if model_variant_key in MODELS else "logits"
    return model_variant_key, default_mode


def get_model_variant_name(model_variant_key: str) -> str:
    """Human-readable name for a model variant."""
    base_key, eval_mode = parse_model_variant_key(model_variant_key)
    if base_key in MODELS:
        return f"{MODELS[base_key].name}-{eval_mode.upper()}"
    return model_variant_key


# ============================================================================
# MAIN EVALUATOR
# ============================================================================

class Evaluator:
    """
    Main evaluation orchestrator.
    
    Coordinates model loading, inference, metrics computation, 
    analysis, and report generation.
    """
    
    def __init__(self, config: Config, batch_size: Optional[int] = None):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration object
            batch_size: Optional override for batch size (uses model default if None)
        """
        self.config = config
        self.model_loader = ModelLoader(config.eval.device)
        self.all_results: Dict[str, Dict[str, Any]] = {}
        self.batch_size_override = batch_size
        
    def run(
        self,
        models: Optional[List[str]] = None,
        skip_inference: bool = False,
        all_eval_types: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run full evaluation pipeline.
        
        Args:
            models: List of model keys to evaluate (None = all)
            skip_inference: Skip inference, load existing predictions
            
        Returns:
            Dict mapping model_key to results
        """
        start_time = time.time()
        
        # Setup
        self._setup()
        
        # Determine model variants to evaluate
        models_to_eval = models or self.config.models_to_evaluate
        model_variants = self._expand_model_variants(models_to_eval)
        model_variants = self._filter_existing_variants(model_variants, all_eval_types=all_eval_types)
        
        print_header(f"VISION MODEL EVALUATION")
        print(f"Model variants to evaluate: {', '.join(model_variants)}")
        print(f"Device: {self.config.eval.device}")
        print(f"Skip inference: {skip_inference}")
        print(f"Force all eval types: {all_eval_types}")

        if not model_variants:
            print("No model variants left to evaluate after skipping existing outputs.")
        
        # Load data
        metadata_df = self._load_data()
        
        # Evaluate each model variant
        for model_variant_key in model_variants:
            try:
                result = self._evaluate_model(
                    model_variant_key=model_variant_key,
                    metadata_df=metadata_df,
                    skip_inference=skip_inference,
                    force_all_eval_types=all_eval_types,
                )
                self.all_results[model_variant_key] = result
            except Exception as e:
                print(f"\n❌ Error evaluating {model_variant_key}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate comparative report
        print("\nLoading all saved models to generate combined comparative report...")
        try:
            self.rebuild_comparative_report()
        except Exception as e:
            print(f"  ⚠ Could not generate comparative report: {e}")
        
        # Save run manifest
        self._save_manifest(model_variants)
        
        total_time = time.time() - start_time
        print_header(f"EVALUATION COMPLETE")
        print(f"Total time: {format_time(total_time)}")
        print(f"Results saved to: {self.config.paths.results_dir}")
        
        return self.all_results

    def _expand_model_variants(self, model_keys: List[str]) -> List[str]:
        """Expand requested model keys into evaluation variants.

        Policy:
        - Run kNN for all models (representation comparability)
        - Run logits for models with native logits mode
        - Run linear probing for models explicitly marked with that extra eval mode
        """
        variants: List[str] = []
        seen = set()

        for model_key in model_keys:
            # Allow explicit variant key passthrough.
            if VARIANT_SEPARATOR in model_key:
                if model_key not in seen:
                    variants.append(model_key)
                    seen.add(model_key)
                continue

            if model_key not in MODELS:
                continue

            spec = MODELS[model_key]

            # Keep native logits mode where available.
            if spec.eval_mode == "logits":
                logits_key = build_model_variant_key(model_key, "logits")
                if logits_key not in seen:
                    variants.append(logits_key)
                    seen.add(logits_key)

            # Add kNN for every model.
            knn_key = build_model_variant_key(model_key, "knn")
            if knn_key not in seen:
                variants.append(knn_key)
                seen.add(knn_key)

            if "linear_probe" in spec.extra_eval_modes:
                linear_probe_key = build_model_variant_key(model_key, "linear_probe")
                if linear_probe_key not in seen:
                    variants.append(linear_probe_key)
                    seen.add(linear_probe_key)

        return variants

    def _filter_existing_variants(
        self,
        model_variants: List[str],
        all_eval_types: bool = False,
    ) -> List[str]:
        """Skip already-evaluated variants unless all eval types are explicitly forced."""
        if all_eval_types:
            return model_variants

        filtered: List[str] = []
        skipped: List[str] = []

        for model_variant_key in model_variants:
            metrics_summary_path = self.config.paths.get_model_output_dir(model_variant_key) / "metrics_summary.json"
            if metrics_summary_path.exists():
                skipped.append(model_variant_key)
                continue
            filtered.append(model_variant_key)

        if skipped:
            print("Skipping already evaluated variants (use --all-eval-types to force rerun):")
            for key in skipped:
                print(f"  - {key}")

        return filtered

    def _load_saved_model_results(
        self,
        models: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Load saved per-model results for the current dataset from disk."""
        dataset_results_dir = self.config.paths.results_dir / self.config.paths.dataset_name

        if not dataset_results_dir.exists():
            raise FileNotFoundError(f"Dataset results directory not found: {dataset_results_dir}")

        requested = set(models or [])
        loaded_results: Dict[str, Dict[str, Any]] = {}

        for metrics_path in sorted(dataset_results_dir.glob("*/metrics_summary.json")):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    report = json.load(f)

                model_key = report.get("model_key", metrics_path.parent.name)
                base_model_key, _ = parse_model_variant_key(model_key)

                if requested and not (
                    model_key in requested
                    or metrics_path.parent.name in requested
                    or base_model_key in requested
                ):
                    continue

                loaded_results[model_key] = {
                    "metrics": report.get("metrics", {}),
                    "stratified": report.get("stratified_analysis", {}),
                    "report_paths": {"json": str(metrics_path)}
                }
            except Exception as e:
                print(f"  ⚠ Skipping {metrics_path}: {e}")

        return loaded_results
    
    def _setup(self):
        """Setup reproducibility and directories."""
        set_seed(self.config.eval.seed)
        set_deterministic(
            self.config.eval.deterministic,
            self.config.eval.benchmark
        )
        ensure_dir(self.config.paths.results_dir)
    
    def _load_data(self):
        """Load metadata and print statistics."""
        print_header("LOADING DATA")
        
        # Load metadata
        print(f"Loading metadata from: {self.config.paths.metadata_path}")
        metadata_df = load_preprocessed_metadata(self.config.paths.metadata_path)
        
        # Get statistics
        stats = get_dataset_statistics(metadata_df)
        print(f"  Total images: {stats['total_images']}")
        print(f"  ShapeNet evaluable: {stats['shapenet_evaluable']} ({stats['shapenet_coverage_pct']:.1f}%)")
        print(f"  Unique ImageNet classes: {stats['unique_imagenet_classes']}")
        print(f"  Unique ShapeNet classes: {stats['unique_shapenet_classes']}")
        
        return metadata_df
    
    def _evaluate_model(
        self,
        model_variant_key: str,
        metadata_df,
        skip_inference: bool = False,
        force_all_eval_types: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model_variant_key: Model variant identifier (e.g., clip_b__knn)
            metadata_df: Loaded metadata DataFrame
            skip_inference: Skip inference and load existing predictions
            
        Returns:
            Dict with all results
        """
        base_model_key, eval_mode = parse_model_variant_key(model_variant_key)
        model_name = get_model_variant_name(model_variant_key)

        print_header(f"EVALUATING: {model_name}")
        
        output_dir = self.config.paths.get_model_output_dir(model_variant_key)
        ensure_dir(output_dir)
        
        predictions_path = output_dir / "predictions.csv"
        
        # Run or load inference
        if skip_inference and predictions_path.exists():
            print("Loading existing predictions...")
            predictions_df = load_predictions(predictions_path)
        else:
            predictions_df = self._run_inference(
                base_model_key,
                eval_mode,
                metadata_df,
                force_retrain=force_all_eval_types,
            )
            save_predictions(predictions_df, predictions_path)
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = compute_metrics_from_predictions(
            predictions_df,
            bootstrap_iterations=self.config.eval.bootstrap_iterations,
            confidence_level=self.config.eval.confidence_level,
            dataset_key=self.config.paths.dataset_key
        )
        
        # Print summary
        self._print_metrics_summary(metrics)
        
        # Run stratified analysis
        print("\nRunning stratified analysis...")
        stratified = run_stratified_analysis(
            predictions_df,
            self.config.eval.stratify_by
        )
        
        # Generate reports
        print("\nGenerating reports...")
        report_paths = generate_model_report(
            model_key=model_variant_key,
            metrics=metrics,
            predictions_df=predictions_df,
            stratified_analysis=stratified,
            output_dir=output_dir
        )
        
        print(f"  ✓ Reports saved to {output_dir}")
        
        # Unload model to free memory
        self.model_loader.unload_model(base_model_key)
        
        return {
            "predictions_path": str(predictions_path),
            "metrics": metrics,
            "stratified": stratified,
            "report_paths": {k: str(v) for k, v in report_paths.items()}
        }
    
    def _run_inference(
        self,
        model_key: str,
        eval_mode: str,
        metadata_df,
        force_retrain: bool = False,
    ) -> pd.DataFrame:
        """Run inference for a model."""
        from .knn import run_knn_inference
        from .linear_probe import run_linear_probe_inference
        
        # Load model
        bundle = self.model_loader.load_model(model_key, override_eval_mode=eval_mode)
        
        # Use batch size override if provided, otherwise use model spec default
        batch_size = self.batch_size_override if self.batch_size_override else bundle.model_spec.batch_size
        
        # Override transforms for PUG dataset (needs specific crop-then-resize zoom) --> otherwise not comparable metrics
        transform = bundle.transform
        if self.config.paths.dataset_key == "pug_imagenet":
            import torchvision.transforms as transforms
            interp_mode = getattr(transforms, "InterpolationMode", None)
            if interp_mode is not None:
                interp = interp_mode.BICUBIC
            else:
                from PIL import Image as PILImage
                interp = PILImage.BICUBIC
                
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            # Try to inherit normalization from the model's standard transform
            if hasattr(bundle.transform, "transforms"):
                for t in bundle.transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        mean = t.mean
                        std = t.std
                        break
            elif hasattr(bundle.transform, "image_processor"):
                # HF image processor fallback
                mean = getattr(bundle.transform.image_processor, "image_mean", mean)
                std = getattr(bundle.transform.image_processor, "image_std", std)
                
            target_size = bundle.model_spec.input_size
            print(f"  Applying PUG-specific transforms (CenterCrop(256) -> Resize({target_size}))")
            transform = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize(target_size, interpolation=interp),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
        # Create dataloader
        # Use configured image root
        image_root = self.config.paths.get_image_root()
        
        dataloader = create_dataloader(
            metadata_df=metadata_df,
            image_root=image_root,
            transform=transform,
            batch_size=batch_size,
            num_workers=self.config.eval.num_workers,
            shuffle=False
        )
        
        print(f"  Dataset size: {len(metadata_df)} images")
        print(f"  Eval mode: {eval_mode}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches: {len(dataloader)}")
        
        # Dispatch based on eval_mode
        if eval_mode == "knn":
            print(f"  Using kNN evaluation (k=20)...")
            predictions_df = run_knn_inference(
                bundle=bundle, 
                dataloader=dataloader, 
                config=self.config,
                k=20
            )
        elif eval_mode == "linear_probe":
            print("  Using linear-probe evaluation...")
            predictions_df = run_linear_probe_inference(
                bundle=bundle,
                dataloader=dataloader,
                config=self.config,
                target_metadata_df=metadata_df,
                force_retrain=force_retrain,
            )
        else:
            # Standard logits-based inference
            predictions_df = run_model_inference(bundle, dataloader, self.config)
        
        return predictions_df
    
    def _print_metrics_summary(self, metrics: Dict[str, Any]):
        """Print metrics summary to console."""
        summary = metrics.get("summary", {})
        img = summary.get("imagenet", {})
        sn = summary.get("shapenet", {})
        
        print("\n  --- Results ---")
        print(f"  ImageNet Top-1:  Base: {img.get('base_top1_accuracy', 0):.1%} |  Normal: {img.get('top1_accuracy', 0):.1%}")
        print(f"  ImageNet Top-5:  Base: {img.get('base_top5_accuracy', 0):.1%} |  Normal: {img.get('top5_accuracy', 0):.1%}")
        
        if not pd.isna(img.get('roc_auc', float('nan'))):
            print(f"  ImageNet ROC AUC: {img.get('roc_auc', 0):.3f}")
            print(f"  ImageNet ECE: {img.get('ece', 0):.3f}")
            
        print(f"  ShapeNet Top-1:  Base: {sn.get('base_top1_accuracy', 0):.1%} |  Normal: {sn.get('top1_accuracy', 0):.1%}  (Evaluated Base/Normal: {sn.get('base_evaluable_samples', 0)}/{sn.get('evaluable_samples', 0)})")
        
        if not pd.isna(sn.get('roc_auc', float('nan'))):
            print(f"  ShapeNet ROC AUC: {sn.get('roc_auc', 0):.3f}")
            print(f"  ShapeNet ECE: {sn.get('ece', 0):.3f}")
    
    def _generate_comparative_report(
        self,
        model_results: Optional[Dict[str, Dict[str, Any]]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """Generate cross-model comparative report."""
        print_header("GENERATING COMPARATIVE REPORT")
        
        model_results = model_results if model_results is not None else self.all_results
        comparative_dir = output_dir or self.config.paths.get_comparative_dir()
        baseline_key = self._resolve_baseline_variant_key(model_results)
        
        report_paths = generate_comparative_report(
            model_results=model_results,
            output_dir=comparative_dir,
            baseline_key=baseline_key
        )
        
        print(f"  ✓ Comparative reports saved to {comparative_dir}")

        return report_paths

    def rebuild_comparative_report(
        self,
        models: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """Rebuild the comparative report from saved runs for the active dataset."""
        print_header("REBUILDING COMPARATIVE REPORT")

        loaded_results = self._load_saved_model_results(models=models)

        if not loaded_results:
            raise ValueError(
                f"No saved model results found under {self.config.paths.results_dir / self.config.paths.dataset_name}"
            )

        print(f"  Loaded {len(loaded_results)} saved model result(s) for {self.config.paths.dataset_name}")
        return self._generate_comparative_report(model_results=loaded_results)

    def _resolve_baseline_variant_key(self, model_results: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """Choose the best matching baseline key in variant-aware runs."""
        results = model_results if model_results is not None else self.all_results
        logits_key = build_model_variant_key(BASELINE_MODEL, "logits")
        knn_key = build_model_variant_key(BASELINE_MODEL, "knn")

        if logits_key in results:
            return logits_key
        if knn_key in results:
            return knn_key
        if BASELINE_MODEL in results:
            return BASELINE_MODEL
        return next(iter(results.keys()), BASELINE_MODEL)
    
    def _save_manifest(self, models_evaluated: List[str]):
        """Save run manifest for reproducibility."""
        manifest = create_run_manifest(
            config=self.config,
            models_evaluated=models_evaluated,
            metadata_path=self.config.paths.metadata_path,
            mapping_path=self.config.paths.shapenet_mapping_path
        )
        
        manifest_path = self.config.paths.results_dir / "run_manifest.json"
        save_json(manifest, manifest_path)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision Model Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=list(MODELS.keys()),
        help=f"Models to evaluate. Available: {', '.join(MODELS.keys())}"
    )
    
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference and use existing predictions"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        help=f"Dataset type to evaluate. Available: {', '.join(DATASETS.keys())}"
    )
    
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to Metadata_Expanded.csv"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for inference (default: use model-specific batch size)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    parser.add_argument(
        "--rebuild-comparative",
        action="store_true",
        help="Rebuild the comparative report from saved runs for the selected dataset and exit"
    )

    parser.add_argument(
        "--all-eval-types",
        action="store_true",
        help="Force rerunning eval variants even if metrics_summary.json already exists"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # List models and exit
    if args.list_models:
        print_available_models()
        return
    
    # Build config
    config = get_default_config()
    
    if args.dataset:
        config.paths.dataset_key = args.dataset
        # Reset the cached paths so they will be repopulated by post_init
        config.paths.image_root = None
        config.paths.metadata_path = None
        config.paths.dataset_name = None
        # Re-trigger post_init logic for proper population of unset optionally typed fields
        config.paths.__post_init__()
    
    if args.metadata:
        config.paths.metadata_path = Path(args.metadata)
    
    if args.output_dir:
        config.paths.results_dir = Path(args.output_dir)
    
    if args.device:
        config.eval.device = args.device

    evaluator = Evaluator(config, batch_size=args.batch_size)

    if args.rebuild_comparative:
        evaluator.rebuild_comparative_report(models=args.models)
        return
    
    # Check prerequisites
    if not config.paths.metadata_path.exists():
        print(f"❌ Error: Metadata file not found: {config.paths.metadata_path}")
        print("   Run preprocess_metadata.py first to create Metadata_Expanded.csv")
        sys.exit(1)
    
    # Run evaluation
    results = evaluator.run(
        models=args.models,
        skip_inference=args.skip_inference,
        all_eval_types=args.all_eval_types,
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RANKINGS (by ShapeNet Top-1 Accuracy)")
    print("=" * 60)
    
    rankings = []
    for model_key, result in results.items():
        sn_acc = result.get("metrics", {}).get("summary", {}).get("shapenet", {}).get("top1_accuracy", 0)
        rankings.append((model_key, sn_acc))
    
    rankings.sort(key=lambda x: -x[1])
    
    for i, (model_key, acc) in enumerate(rankings, 1):
        model_name = get_model_variant_name(model_key)
        baseline_marker = " ⭐" if BASELINE_MODEL in model_key else ""
        print(f"  {i}. {model_name}: {acc:.1%}{baseline_marker}")


if __name__ == "__main__":
    main()
