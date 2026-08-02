"""
k-Nearest Neighbors (kNN) evaluation for Vision Model Evaluation Framework.

Used for models without pretrained ImageNet classification heads (DINOv1, DINOv3).
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
from sklearn.neighbors import KNeighborsClassifier

from .config import Config, UNMAPPED_MARKER
from .models import ModelBundle
from .data_loader import build_reverse_mapping, load_shapenet_mapping
from .utils import load_imagenet_index, format_time


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """
    Extract features from a model for kNN classification.
    """
    
    def __init__(
        self,
        bundle: ModelBundle,
        use_amp: bool = True
    ):
        """
        Initialize feature extractor.
        
        Args:
            bundle: ModelBundle with loaded model (feature extractor mode)
            use_amp: Use automatic mixed precision
        """
        self.bundle = bundle
        self.use_amp = use_amp and torch.cuda.is_available()
        
    @torch.no_grad()
    def extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, Dict[str, List]]:
        """
        Extract features from all images.
        
        Args:
            dataloader: DataLoader with evaluation data
            
        Returns:
            (features, metadata) where features is (N, D) array
        """
        model = self.bundle.model
        device = self.bundle.device
        
        all_features = []
        metadata = {
            "idx": [],
            "image_path": [],
            "true_imagenet_idx": [],
            "true_imagenet_label": [],
            "true_shapenet_superclass": [],
            "shapenet_evaluable": [],
            "Object": [],
            "Level": [],
            "Material": [],
            "Camera Position": [],
            "Light Color (RGB)": [],
            "Fog": [],
        }
        
        model.eval()
        
        for batch in tqdm(dataloader, desc=f"Extracting features ({self.bundle.model_spec.name})"):
            images = batch["images"].to(device)
            
            # Forward pass with optional AMP
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    features = model(images)
            else:
                features = model(images)
                
            # Handle HuggingFace model outputs
            if hasattr(features, "logits"):
                features = features.logits
            elif hasattr(features, "last_hidden_state"):
                # Global average pooling on hidden states if it doesn't return logits
                features = features.last_hidden_state.mean(dim=1)
            
            # Additional processing in case of a tuple
            if isinstance(features, tuple):
                features = features[0]

            # Some models (notably ResNet50 in this setup) return spatial feature maps
            # shaped like (B, C, H, W). Flatten them so kNN always receives 2-D vectors.
            if isinstance(features, torch.Tensor) and features.ndim > 2:
                features = torch.flatten(features, start_dim=1)
            
            # Normalize features
            features = F.normalize(features, p=2, dim=1)
            
            # Move to CPU
            features = features.cpu().numpy()
            all_features.append(features)
            
            # Collect metadata
            batch_size = images.size(0)
            for i in range(batch_size):
                metadata["idx"].append(batch["idx"][i])
                metadata["image_path"].append(batch["image_path"][i])
                metadata["true_imagenet_idx"].append(batch["true_imagenet_idx"][i].item())
                metadata["true_imagenet_label"].append(batch["true_imagenet_label"][i])
                metadata["true_shapenet_superclass"].append(batch["true_shapenet_superclass"][i])
                metadata["shapenet_evaluable"].append(batch["shapenet_evaluable"][i].item())
                
                if "metadata" in batch:
                    meta = batch["metadata"][i]
                    metadata["Object"].append(meta.get("Object", ""))
                    metadata["Level"].append(meta.get("Level", ""))
                    metadata["Material"].append(meta.get("Material", ""))
                    metadata["Camera Position"].append(meta.get("Camera Position", ""))
                    metadata["Light Color (RGB)"].append(meta.get("Light Color (RGB)", ""))
                    metadata["Fog"].append(meta.get("Fog", ""))
        
        features = np.vstack(all_features)
        print(f"  Extracted features: {features.shape}")
        
        return features, metadata


# ============================================================================
# KNN CLASSIFIER
# ============================================================================

class KNNClassifier:
    """
    k-Nearest Neighbors classifier for image classification.
    """
    
    def __init__(
        self,
        k: int = 20,
        temperature: float = 0.07,
        metric: str = "cosine"
    ):
        """
        Initialize kNN classifier.
        
        Args:
            k: Number of neighbors
            temperature: Temperature for softmax (lower = sharper)
            metric: Distance metric
        """
        self.k = k
        self.temperature = temperature
        self.metric = metric
        self.knn = None
        self.train_labels = None
        
    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Fit kNN on training features.
        
        For zero-shot evaluation, we use the full dataset as both
        train and test (leave-one-out style for fair comparison).
        
        Args:
            features: (N, D) feature array
            labels: (N,) label array
        """
        self.knn = KNeighborsClassifier(
            n_neighbors=self.k,
            metric=self.metric,
            algorithm="auto",
            weights="distance"
        )
        self.knn.fit(features, labels)
        self.train_labels = labels
        
    def predict(
        self,
        features: np.ndarray,
        top_k: int = 5,
        sample_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for query features.
        
        Args:
            features: (N, D) query features
            top_k: Number of top predictions to return
            sample_indices: (N,) array of sample indices for leave-one-out
            
        Returns:
            (top_k_labels, top_k_probs)
        """
        # Get k neighbors
        distances, indices = self.knn.kneighbors(features)
        
        # Get neighbor labels
        neighbor_labels = self.train_labels[indices]  # (N, k)
        
        n_samples = features.shape[0]
        unique_labels = np.unique(self.train_labels)
        n_classes = len(unique_labels)
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        
        votes = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            for j in range(self.k):
                neighbor_idx = indices[i, j]
                
                # Leave-one-out: skip if this neighbor is the sample itself
                if sample_indices is not None and neighbor_idx == sample_indices[i]:
                    continue
                
                label = neighbor_labels[i, j]
                class_idx = label_to_idx[label]
                weight = 1.0 / (distances[i, j] + 1e-8)
                votes[i, class_idx] += weight
        
        # Softmax to get probabilities
        votes = votes / self.temperature
        probs = np.exp(votes - votes.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        # Get top-k
        top_k_idx = np.argsort(-probs, axis=1)[:, :top_k]
        top_k_probs = np.take_along_axis(probs, top_k_idx, axis=1)
        top_k_labels = unique_labels[top_k_idx]
        
        return top_k_labels, top_k_probs


# ============================================================================
# KNN INFERENCE
# ============================================================================

def run_knn_inference(
    bundle: ModelBundle,
    dataloader: DataLoader,
    config: Config,
    k: int = 20
) -> pd.DataFrame:
    """
    Run kNN-based inference for models without classification heads.
    
    Uses leave-one-out style: fit on all samples, predict each sample
    using its k nearest neighbors (excluding itself).
    
    Args:
        bundle: Loaded model bundle (feature extractor)
        dataloader: DataLoader with images
        config: Configuration
        k: Number of neighbors
        
    Returns:
        DataFrame with predictions (same format as logits inference)
    """
    print(f"\nRunning kNN evaluation (k={k})...")
    start_time = time.time()
    
    # Load mappings
    imagenet_index = load_imagenet_index(config.paths.imagenet_index_path)
    shapenet_mapping = load_shapenet_mapping(config.paths.shapenet_mapping_path)
    imagenet_to_shapenet = build_reverse_mapping(shapenet_mapping)
    
    # Extract features
    extractor = FeatureExtractor(bundle, use_amp=config.eval.use_amp)
    features, metadata = extractor.extract_features(dataloader)
    
    # Get labels
    labels = np.array(metadata["true_imagenet_idx"])
    n_samples = features.shape[0]
    
    # Create sample indices for leave-one-out
    sample_indices = np.arange(n_samples)
    
    # Fit kNN (using all data - we'll do leave-one-out prediction)
    print(f"  Fitting kNN classifier (k={k})...")
    knn = KNNClassifier(k=k + 1)  # +1 because we'll exclude self
    knn.fit(features, labels)
    
    # Predict with leave-one-out (pass sample_indices)
    print("  Predicting...")
    top_k_labels, top_k_probs = knn.predict(
        features, 
        top_k=config.eval.top_k,
        sample_indices=sample_indices
    )
    
    # Get actual number of classes (may be less than top_k)
    n_classes_available = top_k_labels.shape[1]
    
    # Build results DataFrame
    results = []
    
    for i in range(n_samples):
        true_idx = metadata["true_imagenet_idx"][i]
        
        result = {
            "idx": metadata["idx"][i],
            "image_path": metadata["image_path"][i],
            "true_imagenet_idx": true_idx,
            "true_imagenet_label": metadata["true_imagenet_label"][i],
            "true_shapenet_superclass": metadata["true_shapenet_superclass"][i],
            "shapenet_evaluable": metadata["shapenet_evaluable"][i],
        }
        
        # Add top-K predictions (pad with -1 if fewer classes available)
        preds = []
        for k_idx in range(config.eval.top_k):
            if k_idx < n_classes_available:
                pred_idx = int(top_k_labels[i, k_idx])
                pred_prob = float(top_k_probs[i, k_idx])
            else:
                pred_idx = -1
                pred_prob = 0.0
            
            pred_label = imagenet_index.get(pred_idx, f"<unknown:{pred_idx}>") if pred_idx >= 0 else "<none>"
            
            result[f"top{k_idx+1}_idx"] = pred_idx
            result[f"top{k_idx+1}_label"] = pred_label
            result[f"top{k_idx+1}_prob"] = pred_prob
            preds.append(pred_idx)
        
        # Map top-1 prediction to ShapeNet
        top1_idx = int(top_k_labels[i, 0])
        pred_shapenet = imagenet_to_shapenet.get(top1_idx, UNMAPPED_MARKER)
        result["pred_shapenet_top1"] = pred_shapenet
        
        # Compute correctness flags
        result["imagenet_top1_correct"] = (top1_idx == true_idx)
        result["imagenet_top5_correct"] = (true_idx in preds[:5])
        
        # ShapeNet correctness
        true_shapenet = metadata["true_shapenet_superclass"][i]
        if metadata["shapenet_evaluable"][i] and pred_shapenet != UNMAPPED_MARKER:
            result["shapenet_top1_correct"] = (pred_shapenet == true_shapenet)
        else:
            result["shapenet_top1_correct"] = None
        
        # Add metadata
        result["Object"] = metadata["Object"][i]
        result["Level"] = metadata["Level"][i]
        result["Material"] = metadata["Material"][i]
        result["Camera Position"] = metadata["Camera Position"][i]
        result["Light Color (RGB)"] = metadata["Light Color (RGB)"][i]
        result["Fog"] = metadata["Fog"][i]
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    elapsed = time.time() - start_time
    print(f"  kNN evaluation completed in {format_time(elapsed)}")
    
    # Quick accuracy check
    top1_acc = df["imagenet_top1_correct"].mean()
    print(f"  ImageNet Top-1 Accuracy: {top1_acc:.1%}")
    
    return df
