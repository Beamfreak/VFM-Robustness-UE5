"""
Model loading and management for Vision Model Evaluation Framework.

Uses timm to load pretrained models for ImageNet classification.
Supports HuggingFace transformers for specific models (DINOv2-IN1K).
"""

from dataclasses import dataclass, replace
from typing import Optional, Callable, Any

import torch
import torch.nn as nn
import timm
from timm.data import create_transform, resolve_data_config

from .config import ModelSpec, MODELS


def _is_huggingface_model(model_id: str) -> bool:
    """Check if model ID is a HuggingFace model."""
    return model_id.startswith("facebook/") or "/" in model_id and not model_id.startswith("timm/")


# ============================================================================
# MODEL BUNDLE
# ============================================================================

@dataclass
class ModelBundle:
    """
    Container for a loaded model and its components.
    
    Attributes:
        model: PyTorch model
        transform: Image preprocessing transform
        device: Target device
        model_key: Model identifier key
        model_spec: Model specification
    """
    model: nn.Module
    transform: Callable
    device: torch.device
    model_key: str
    model_spec: ModelSpec
    
    def to(self, device: torch.device) -> "ModelBundle":
        """Move model to device."""
        self.model = self.model.to(device)
        self.device = device
        return self


# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """
    Load and manage foundation models via timm.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize model loader.
        
        Args:
            device: Target device ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._loaded_models = {}
    
    def load_model(self, model_key: str, override_eval_mode: Optional[str] = None) -> ModelBundle:
        """
        Load a model by its key.
        
        Args:
            model_key: Key from MODELS dict (e.g., "clip_b", "resnet50")
            
        Returns:
            ModelBundle with model, transforms, and metadata
        """
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
        
        spec = MODELS[model_key]
        eval_mode = override_eval_mode or spec.eval_mode
        if eval_mode not in {"logits", "knn", "linear_probe"}:
            raise ValueError(f"Unsupported eval_mode override: {eval_mode}")

        if eval_mode != spec.eval_mode:
            spec = replace(spec, eval_mode=eval_mode)

        cache_key = f"{model_key}__{eval_mode}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
        
        print(f"Loading {spec.name} ({spec.model_id})...")
        
        # Check if this is a HuggingFace model
        if _is_huggingface_model(spec.model_id):
            bundle = self._load_huggingface_model(model_key, spec)
        else:
            # Load via timm
            # For feature-based evaluation modes, load as feature extractor (num_classes=0)
            num_classes = 0 if spec.eval_mode in {"knn", "linear_probe"} else 1000
            
            # Explicitly force CLS token pooling for DINO models as standard 'avg'
            # pooling destroys representations in the newly added DINOv3 in timm
            kwargs = {}
            if "dinov3" in spec.model_id:
                kwargs["global_pool"] = "token"
            
            model = timm.create_model(
                spec.model_id,
                pretrained=True,
                num_classes=num_classes,
                **kwargs
            )
            
            model.eval()
            model = model.to(self.device)
            
            # Get appropriate transforms from timm
            data_config = resolve_data_config({}, model=model)
            
            # Override input size if specified in spec
            if spec.input_size:
                data_config['input_size'] = (3, spec.input_size, spec.input_size)
            
            transform = create_transform(**data_config, is_training=False)
            
            bundle = ModelBundle(
                model=model,
                transform=transform,
                device=self.device,
                model_key=model_key,
                model_spec=spec
            )
        
        self._loaded_models[cache_key] = bundle
        print(f"  ✓ Loaded on {self.device} (eval_mode={spec.eval_mode})")
        
        return bundle
    
    def _load_huggingface_model(self, model_key: str, spec: ModelSpec) -> ModelBundle:
        """
        Load a HuggingFace transformers model (e.g., DINOv2 with ImageNet head).
        """
        try:
            from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification
            import torchvision.transforms as T
        except ImportError:
            raise ImportError(
                "HuggingFace transformers required for this model. "
                "Install with: pip install transformers"
            )

        model_loader_cls = AutoModel if spec.eval_mode in {"knn", "linear_probe"} else AutoModelForImageClassification

        def load_model_and_processor() -> tuple[nn.Module, Any]:
            try:
                return (
                    model_loader_cls.from_pretrained(spec.model_id),
                    AutoImageProcessor.from_pretrained(spec.model_id),
                )
            except ValueError as e:
                try:
                    return (
                        model_loader_cls.from_pretrained(spec.model_id, trust_remote_code=True),
                        AutoImageProcessor.from_pretrained(spec.model_id, trust_remote_code=True),
                    )
                except Exception:
                    try:
                        from huggingface_hub import hf_hub_download
                        import json

                        cfg_path = hf_hub_download(repo_id=spec.model_id, filename="config.json")
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        if "model_type" not in cfg:
                            raise ValueError(
                                f"Model repo `{spec.model_id}` config.json is missing 'model_type'. "
                                "This prevents `transformers` from auto-detecting the architecture. "
                                "Possible fixes: upgrade `transformers` to the latest version, set "
                                "`trust_remote_code=True` (already attempted), or use a model repo that includes a standard `config.json` with `model_type`."
                            )
                    except Exception:
                        raise ValueError(
                            f"Unrecognized model in {spec.model_id}. Should have a `model_type` key in its config.json. "
                            "Tried loading with and without `trust_remote_code=True`."
                        ) from e

                    raise
        
        # Load model and processor. Some HF repos use custom code/configs
        # and do not include a standard `model_type` in config.json —
        # in that case retry with `trust_remote_code=True`.
        loaded_model, loaded_processor = load_model_and_processor()
        
        loaded_model.eval()
        model = loaded_model.to(self.device)
        
        # Create transform from processor config
        input_size = spec.input_size or 224
        
        # Standard ImageNet normalization
        normalize = T.Normalize(
            mean=loaded_processor.image_mean if hasattr(loaded_processor, 'image_mean') else [0.485, 0.456, 0.406],
            std=loaded_processor.image_std if hasattr(loaded_processor, 'image_std') else [0.229, 0.224, 0.225]
        )
        
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            normalize
        ])
        
        return ModelBundle(
            model=model,
            transform=transform,
            device=self.device,
            model_key=model_key,
            model_spec=spec
        )
    
    def unload_model(self, model_key: str):
        """
        Unload a model to free memory.
        
        Args:
            model_key: Model key to unload
        """
        keys_to_remove = [
            key for key in self._loaded_models.keys()
            if key == model_key or key.startswith(f"{model_key}__")
        ]
        for key in keys_to_remove:
            del self._loaded_models[key]
        if keys_to_remove:
            torch.cuda.empty_cache()
    
    def unload_all(self):
        """Unload all models."""
        self._loaded_models.clear()
        torch.cuda.empty_cache()
    
    def get_model_info(self, model_key: str) -> dict:
        """
        Get information about a model.
        
        Args:
            model_key: Model key
            
        Returns:
            Dict with model information
        """
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        
        spec = MODELS[model_key]
        
        info = {
            "key": model_key,
            "name": spec.name,
            "model_id": spec.model_id,
            "input_size": spec.input_size,
            "batch_size": spec.batch_size,
            "precision": spec.precision,
            "is_loaded": model_key in self._loaded_models,
        }
        
        if model_key in self._loaded_models:
            bundle = self._loaded_models[model_key]
            info["device"] = str(bundle.device)
            info["num_params"] = sum(p.numel() for p in bundle.model.parameters())
        
        return info


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_models() -> dict:
    """
    Get information about all available models.
    
    Returns:
        Dict mapping model_key to model info
    """
    result = {}
    for key, spec in MODELS.items():
        result[key] = {
            "name": spec.name,
            "model_id": spec.model_id,
            "input_size": spec.input_size,
            "batch_size": spec.batch_size,
        }
    return result


def print_available_models():
    """Print table of available models."""
    print("\nAvailable Models:")
    print("-" * 80)
    print(f"{'Key':<12} {'Name':<15} {'Model ID':<40} {'Size':<6} {'Batch':<6}")
    print("-" * 80)
    for key, spec in MODELS.items():
        print(f"{key:<12} {spec.name:<15} {spec.model_id:<40} {spec.input_size:<6} {spec.batch_size:<6}")
    print("-" * 80)


def verify_model_availability():
    """
    Verify all models can be loaded (dry run).
    
    Prints status for each model.
    """
    print("\nVerifying model availability...")
    for key in MODELS:
        try:
            # Just check if model exists in timm registry
            spec = MODELS[key]
            available = spec.model_id in timm.list_models(pretrained=True)
            if available:
                print(f"  ✓ {spec.name}: Available")
            else:
                # Try with wildcard
                matches = timm.list_models(f"*{spec.model_id.split('.')[0]}*", pretrained=True)
                if matches:
                    print(f"  ⚠ {spec.name}: Using similar model (found {len(matches)} variants)")
                else:
                    print(f"  ✗ {spec.name}: NOT FOUND in timm")
        except Exception as e:
            print(f"  ✗ {key}: Error - {e}")
