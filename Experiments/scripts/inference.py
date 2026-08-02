"""
Inference engine for Vision Model Evaluation Framework.

Runs batch inference on all images and collects top-K predictions.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, UNMAPPED_MARKER
from .models import ModelBundle
from .data_loader import build_reverse_mapping, load_shapenet_mapping
from .utils import load_imagenet_index, format_time, optimize_dataframe_types


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """
    Run inference on a model and collect predictions.
    """
    
    def __init__(
        self,
        bundle: ModelBundle,
        imagenet_index: Dict[int, str],
        imagenet_to_shapenet: Dict[int, str],
        top_k: int = 5,
        use_amp: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            bundle: ModelBundle with loaded model
            imagenet_index: Mapping of ImageNet index to label
            imagenet_to_shapenet: Mapping of ImageNet index to ShapeNet superclass
            top_k: Number of top predictions to store
            use_amp: Use automatic mixed precision
        """
        self.bundle = bundle
        self.imagenet_index = imagenet_index
        self.imagenet_to_shapenet = imagenet_to_shapenet
        self.top_k = top_k
        self.use_amp = use_amp and torch.cuda.is_available()
        
    @torch.no_grad()
    def run_inference(self, dataloader: DataLoader) -> pd.DataFrame:
        """
        Run inference on all batches.
        
        Args:
            dataloader: DataLoader with evaluation data
            
        Returns:
            DataFrame with all predictions
        """
        model = self.bundle.model
        device = self.bundle.device
        
        all_results = []
        total_time = 0.0
        
        model.eval()
        
        for batch in tqdm(dataloader, desc=f"Inference ({self.bundle.model_spec.name})"):
            start_time = time.time()
            
            images = batch["images"].to(device)
            
            # Forward pass with optional AMP
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    output = model(images)
            else:
                output = model(images)
            
            # Handle HuggingFace vs timm output format
            if hasattr(output, 'logits'):
                # HuggingFace model (e.g., DINOv2-ImageNet)
                logits = output.logits
            else:
                # timm model (direct tensor output)
                logits = output
            
            # Get probabilities and top-K predictions
            probs = F.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=1)
            
            # Move to CPU
            topk_probs = topk_probs.cpu().numpy()
            topk_indices = topk_indices.cpu().numpy()
            
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Process each sample in batch
            batch_size = images.size(0)
            for i in range(batch_size):
                result = self._process_sample(
                    batch=batch,
                    sample_idx=i,
                    topk_indices=topk_indices[i],
                    topk_probs=topk_probs[i]
                )
                all_results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        df = optimize_dataframe_types(df)
        
        print(f"  Inference completed in {format_time(total_time)}")
        if total_time > 0:
            print(f"  Throughput: {len(df) / total_time:.1f} images/sec")
        else:
            print("  Throughput: N/A")
        
        return df
    
    def _process_sample(
        self,
        batch: Dict,
        sample_idx: int,
        topk_indices: np.ndarray,
        topk_probs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a single sample and create result record.
        
        Args:
            batch: Batch dict from dataloader
            sample_idx: Index within batch
            topk_indices: Top-K predicted indices
            topk_probs: Top-K predicted probabilities
            
        Returns:
            Dict with all prediction fields
        """
        # Ground truth
        true_idx = batch["true_imagenet_idx"][sample_idx].item()
        true_label = batch["true_imagenet_label"][sample_idx]
        true_shapenet = batch["true_shapenet_superclass"][sample_idx]
        shapenet_evaluable = batch["shapenet_evaluable"][sample_idx].item()
        
        # Build result
        result = {
            "idx": batch["idx"][sample_idx],
            "image_path": batch["image_path"][sample_idx],
            "true_imagenet_idx": true_idx,
            "true_imagenet_label": true_label,
            "true_shapenet_superclass": true_shapenet,
            "shapenet_evaluable": shapenet_evaluable,
        }
        
        # Add top-K predictions
        for k in range(self.top_k):
            pred_idx = int(topk_indices[k])
            pred_label = self.imagenet_index.get(pred_idx, f"<unknown:{pred_idx}>")
            pred_prob = float(topk_probs[k])
            
            result[f"top{k+1}_idx"] = pred_idx
            result[f"top{k+1}_label"] = pred_label
            result[f"top{k+1}_prob"] = pred_prob
        
        # Map top-1 prediction to ShapeNet
        top1_idx = int(topk_indices[0])
        pred_shapenet = self.imagenet_to_shapenet.get(top1_idx, UNMAPPED_MARKER)
        result["pred_shapenet_top1"] = pred_shapenet
        
        # Compute correctness flags
        if batch.get("valid_imagenet_indices") and len(batch["valid_imagenet_indices"]) > sample_idx:
            valid_indices_str = batch["valid_imagenet_indices"][sample_idx]
            valid_indices = [int(x) for x in valid_indices_str.split(',')]
            result["imagenet_top1_correct"] = (top1_idx in valid_indices)
            result["imagenet_top5_correct"] = any(idx in valid_indices for idx in topk_indices[:5])
        else:
            result["imagenet_top1_correct"] = (top1_idx == true_idx)
            result["imagenet_top5_correct"] = (true_idx in topk_indices[:5])
        
        # ShapeNet correctness (only if evaluable)
        if shapenet_evaluable and pred_shapenet != UNMAPPED_MARKER:
            result["shapenet_top1_correct"] = (pred_shapenet == true_shapenet)
        else:
            result["shapenet_top1_correct"] = None
        
        # Add metadata
        if "metadata" in batch:
            meta = batch["metadata"][sample_idx]
            result["Object"] = meta.get("Object", "")
            result["Level"] = meta.get("Level", "")
            result["Material"] = meta.get("Material", "")
            result["Camera Position"] = meta.get("Camera Position", "")
            result["Light Color (RGB)"] = meta.get("Light Color (RGB)", "")
            result["Fog"] = meta.get("Fog", "")
        
        return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_model_inference(
    bundle: ModelBundle,
    dataloader: DataLoader,
    config: Config
) -> pd.DataFrame:
    """
    Run inference for a single model.
    
    Args:
        bundle: Loaded model bundle
        dataloader: DataLoader with images
        config: Configuration
        
    Returns:
        DataFrame with predictions
    """
    # Load mappings
    imagenet_index = load_imagenet_index(config.paths.imagenet_index_path)
    shapenet_mapping = load_shapenet_mapping(config.paths.shapenet_mapping_path)
    imagenet_to_shapenet = build_reverse_mapping(shapenet_mapping)
    
    # Create engine and run
    engine = InferenceEngine(
        bundle=bundle,
        imagenet_index=imagenet_index,
        imagenet_to_shapenet=imagenet_to_shapenet,
        top_k=config.eval.top_k,
        use_amp=config.eval.use_amp
    )
    
    predictions_df = engine.run_inference(dataloader)
    
    return predictions_df


def save_predictions(
    predictions_df: pd.DataFrame,
    output_path: Path,
    separator: str = ";"
):
    """
    Save predictions to CSV.
    
    Args:
        predictions_df: Predictions DataFrame
        output_path: Path to save CSV
        separator: CSV separator
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, sep=separator, index=False)
    print(f"  Saved predictions to {output_path}")


def load_predictions(predictions_path: Path, separator: str = ";") -> pd.DataFrame:
    """
    Load predictions from CSV.
    
    Args:
        predictions_path: Path to predictions CSV
        separator: CSV separator
        
    Returns:
        Predictions DataFrame
    """
    df = pd.read_csv(predictions_path, sep=separator)
    return optimize_dataframe_types(df)
