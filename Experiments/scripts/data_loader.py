"""
Data loading utilities for Vision Model Evaluation Framework.

Handles loading preprocessed metadata and creating PyTorch dataloaders.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .config import METADATA_COLUMNS, UNMAPPED_MARKER, PathConfig
from .utils import check_image_exists, load_imagenet_index


# ============================================================================
# DATASET CLASS
# ============================================================================

class EvaluationDataset(Dataset):
    """
    PyTorch Dataset for evaluation images.
    
    Loads images from preprocessed metadata and applies transforms.
    """
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_root: Path,
        transform=None,
        return_metadata: bool = True,
        validate_images: str = "sample"
    ):
        """
        Initialize dataset.
        
        Args:
            metadata_df: DataFrame with preprocessed metadata
            image_root: Root directory containing images
            transform: Image transform to apply
            return_metadata: Include metadata in __getitem__ output
            validate_images: How to validate image existence ("all", "sample", "none")
        """
        self.image_root = Path(image_root)
        self.transform = transform
        self.return_metadata = return_metadata
        
        # Keep temporary reference to df for validation and extraction
        self.metadata_df = metadata_df.reset_index(drop=True)
        
        # Verify image paths exist
        self.valid_indices = self._validate_images(validate_images)
        if len(self.valid_indices) == 0:
            raise RuntimeError(
                f"No valid images found in {self.image_root} "
                "matching the metadata. Check paths."
            )
            
        self.original_indices = self.valid_indices
        
        # Extract columns as lists to free the pandas DataFrame (preventing COW copy issue in workers)
        df_valid = self.metadata_df.iloc[self.valid_indices].reset_index(drop=True)
        
        self.images = df_valid["Image"].tolist()
        self.classes = df_valid["Class"].astype(int).tolist()
        self.labels = df_valid["ImageNet_Label"].tolist()
        self.shapenet_superclasses = df_valid["ShapeNet_Superclass"].tolist()
        
        if "Valid_Classes" in df_valid.columns:
            self.valid_classes = df_valid["Valid_Classes"].tolist()
        else:
            self.valid_classes = None
            
        if self.return_metadata:
            self.objects = df_valid.get("Object", pd.Series("", index=df_valid.index)).tolist()
            self.levels = df_valid.get("Level", pd.Series("", index=df_valid.index)).tolist()
            self.materials = df_valid.get("Material", pd.Series("", index=df_valid.index)).tolist()
            self.camera_positions = df_valid.get("Camera Position", pd.Series("", index=df_valid.index)).tolist()
            self.light_colors = df_valid.get("Light Color (RGB)", pd.Series("", index=df_valid.index)).tolist()
            self.fogs = df_valid.get("Fog", pd.Series("", index=df_valid.index)).tolist()
            
        # Free memory by deleting the metadata DataFrame
        self.metadata_df = None
        import gc
        gc.collect()
        
    def _validate_images(self, mode: str = "sample") -> List[int]:
        """Check which images exist and are loadable."""
        if mode == "none":
            return list(range(len(self.metadata_df)))
            
        if mode == "sample":
            n_samples = len(self.metadata_df)
            if n_samples == 0:
                return []
                
            # Deterministic sample of up to 100 images
            step = max(1, n_samples // 100)
            sample_indices = list(range(0, n_samples, step))[:100]
            
            all_exist = True
            for idx in sample_indices:
                rel_path = self.metadata_df.loc[idx, "Image"]
                full_path = self.image_root / rel_path
                if not full_path.exists():
                    all_exist = False
                    break
            if all_exist:
                return list(range(n_samples))
                
            print("  [Warning] Some sample images not found. Performing full validation...")
            
        valid = []
        for idx in range(len(self.metadata_df)):
            rel_path = self.metadata_df.loc[idx, "Image"]
            full_path = self.image_root / rel_path
            if full_path.exists():
                valid.append(idx)
        return valid
    
    def __len__(self) -> int:
        return len(self.original_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item by index.
        """
        real_idx = self.original_indices[idx]
        
        # Load image
        rel_path = self.images[idx]
        full_path = self.image_root / rel_path
        image = Image.open(full_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        result = {
            "image": image,
            "idx": real_idx,
            "image_path": rel_path,
            "true_imagenet_idx": self.classes[idx],
            "true_imagenet_label": self.labels[idx],
            "true_shapenet_superclass": self.shapenet_superclasses[idx],
            "shapenet_evaluable": self.shapenet_superclasses[idx] != UNMAPPED_MARKER,
        }
        
        if self.valid_classes is not None:
            val_cls = self.valid_classes[idx]
            if pd.notna(val_cls):
                result["valid_imagenet_indices"] = str(val_cls)
        
        if self.return_metadata:
            result["metadata"] = {
                "Object": self.objects[idx],
                "Level": self.levels[idx],
                "Material": self.materials[idx],
                "Camera Position": self.camera_positions[idx],
                "Light Color (RGB)": self.light_colors[idx],
                "Fog": self.fogs[idx],
            }
        
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader.
    
    Stacks images into batch tensor and collects other fields as lists.
    """
    images = torch.stack([item["image"] for item in batch])
    
    result = {
        "images": images,
        "idx": [item["idx"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "true_imagenet_idx": torch.tensor([item["true_imagenet_idx"] for item in batch]),
        "true_imagenet_label": [item["true_imagenet_label"] for item in batch],
        "true_shapenet_superclass": [item["true_shapenet_superclass"] for item in batch],
        "shapenet_evaluable": torch.tensor([item["shapenet_evaluable"] for item in batch]),
    }
    
    if "valid_imagenet_indices" in batch[0]:
        result["valid_imagenet_indices"] = [item.get("valid_imagenet_indices", "") for item in batch]
        
    if "metadata" in batch[0]:
        result["metadata"] = [item["metadata"] for item in batch]
    
    return result


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_preprocessed_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load preprocessed metadata CSV.
    
    Args:
        metadata_path: Path to Metadata_Expanded.csv
        
    Returns:
        DataFrame with all metadata
    """
    # Peek at the columns first to build dtype dictionary for memory optimization
    header_df = pd.read_csv(metadata_path, sep=";", nrows=0)
    columns = header_df.columns
    
    dtypes = {}
    for col in columns:
        if col == "Class":
            dtypes[col] = "int16"
        elif col in ["Image", "Valid_Classes"]:
            dtypes[col] = "str"
        else:
            dtypes[col] = "category"
            
    df = pd.read_csv(metadata_path, sep=";", dtype=dtypes)
    
    # Filter out segmentation mask images
    if "Image" in df.columns:
        df = df[~df["Image"].str.contains("_mask", case=False, na=False)].reset_index(drop=True)
        # Clean UE absolute paths (e.g. ../../Dataset/Camera/... -> Camera/...)
        df["Image"] = df["Image"].apply(
            lambda x: x.split("Dataset/")[-1] if isinstance(x, str) and "Dataset/" in x else x
        )
    
    # Validate required columns
    required = ["Image", "Class", "ImageNet_Label", "ShapeNet_Superclass"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in metadata: {missing}. "
            "Did you run preprocess_metadata.py first?"
        )
    
    # Ensure Class is integer
    df["Class"] = df["Class"].astype(int)

    # Some external ImageNet-D preprocessing pipelines store local 0..89 IDs.
    # Remap to global ImageNet-1k IDs so top-k metrics are computed correctly.
    df = _maybe_remap_imagenet_d_labels(df, metadata_path)
    
    return df


def _normalize_label(text: str) -> str:
    """Normalize class labels for robust matching."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_imagenet_d_category(image_path: str) -> Optional[str]:
    """Extract semantic category from ImageNet-D relative path."""
    if not isinstance(image_path, str):
        return None
    parts = image_path.split("/")
    if len(parts) < 2:
        return None
    if parts[0] in {"background", "material", "texture"}:
        return parts[1]
    return None


def _load_imagenet_d_category_to_label(metadata_path: Path) -> Dict[str, str]:
    """Load category -> human-readable ground-truth label from ImageNet-D questions files."""
    questions_dir = metadata_path.parent / "questions"
    if not questions_dir.exists():
        return {}

    mapping: Dict[str, str] = {}
    for csv_path in questions_dir.glob("*.csv"):
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("image_dir"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                image_rel = parts[0]
                gt_label = parts[1].strip()
                cat = _extract_imagenet_d_category(image_rel)
                if cat and gt_label:
                    mapping[cat] = gt_label
    return mapping


def _maybe_remap_imagenet_d_labels(df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    """Remap ImageNet-D local class IDs to global ImageNet-1k IDs when detected."""
    looks_like_imagenet_d = "imagenet-d" in str(metadata_path).lower()
    looks_local = (
        len(df) > 0
        and df["Class"].min() >= 0
        and df["Class"].max() < 200
        and bool(df["ImageNet_Label"].astype(str).str.fullmatch(r"class_\d+").all())
    )

    if not looks_like_imagenet_d and not looks_local:
        return df

    category_to_label = _load_imagenet_d_category_to_label(metadata_path)
    if not category_to_label:
        return df

    try:
        repo_root = Path(__file__).resolve().parent.parent
        imagenet_index_path = repo_root / "imagenet_class_index.txt"
        idx_to_label = load_imagenet_index(imagenet_index_path)
    except Exception:
        return df

    normalized_label_to_idx = {
        _normalize_label(label): idx
        for idx, label in idx_to_label.items()
    }

    new_classes: List[int] = []
    new_labels: List[str] = []
    remapped = 0

    for _, row in df.iterrows():
        image_path = row["Image"]
        category = _extract_imagenet_d_category(image_path)
        gt_label = category_to_label.get(category, "") if category else ""
        gt_idx = normalized_label_to_idx.get(_normalize_label(gt_label))

        if gt_idx is None:
            new_classes.append(int(row["Class"]))
            new_labels.append(row["ImageNet_Label"])
            continue

        new_classes.append(int(gt_idx))
        new_labels.append(idx_to_label[int(gt_idx)])
        remapped += 1

    if remapped > 0:
        df = df.copy()
        df["Class"] = new_classes
        df["ImageNet_Label"] = new_labels
        print(f"  Detected ImageNet-D local labels; remapped {remapped}/{len(df)} samples to ImageNet-1k indices.")

    return df


def create_dataloader(
    metadata_df: pd.DataFrame,
    image_root: Path,
    transform,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = False,
    validate_images: str = "sample"
) -> DataLoader:
    """
    Create DataLoader for evaluation.
    
    Args:
        metadata_df: Preprocessed metadata DataFrame
        image_root: Root directory for images
        transform: Image transforms to apply
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        validate_images: How to validate image existence ("all", "sample", "none")
        
    Returns:
        DataLoader instance
    """
    dataset = EvaluationDataset(
        metadata_df=metadata_df,
        image_root=image_root,
        transform=transform,
        return_metadata=True,
        validate_images=validate_images
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    return loader


# ============================================================================
# MAPPING UTILITIES
# ============================================================================

def load_shapenet_mapping(mapping_path: Path) -> Dict[str, List[int]]:
    """
    Load ShapeNet to ImageNet mapping.
    
    Args:
        mapping_path: Path to mapping JSON
        
    Returns:
        Dict mapping ShapeNet superclass to list of ImageNet indices
    """
    import json
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    result = {}
    for category, payload in mapping.items():
        indices = payload.get("imagenet_class_indices", [])
        result[category] = [int(i) for i in indices]
    
    return result


def build_reverse_mapping(shapenet_mapping: Dict[str, List[int]]) -> Dict[int, str]:
    """
    Build reverse mapping from ImageNet index to ShapeNet superclass.
    
    Args:
        shapenet_mapping: ShapeNet to ImageNet mapping
        
    Returns:
        Dict mapping ImageNet index to ShapeNet superclass
    """
    reverse = {}
    for category, indices in shapenet_mapping.items():
        for idx in indices:
            reverse[idx] = category
    return reverse


def get_shapenet_categories(metadata_df: pd.DataFrame) -> List[str]:
    """
    Get list of ShapeNet superclasses present in dataset.
    
    Args:
        metadata_df: Metadata DataFrame
        
    Returns:
        Sorted list of unique ShapeNet superclasses (excluding unmapped)
    """
    categories = metadata_df["ShapeNet_Superclass"].unique().tolist()
    categories = [c for c in categories if c != UNMAPPED_MARKER]
    return sorted(categories)


def get_dataset_statistics(metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic dataset statistics.
    
    Args:
        metadata_df: Metadata DataFrame
        
    Returns:
        Dict with statistics
    """
    total = len(metadata_df)
    shapenet_evaluable = (metadata_df["ShapeNet_Superclass"] != UNMAPPED_MARKER).sum()
    
    stats = {
        "total_images": total,
        "shapenet_evaluable": int(shapenet_evaluable),
        "shapenet_not_evaluable": total - int(shapenet_evaluable),
        "shapenet_coverage_pct": 100 * shapenet_evaluable / total if total > 0 else 0,
        "unique_imagenet_classes": metadata_df["Class"].nunique(),
        "unique_shapenet_classes": metadata_df[metadata_df["ShapeNet_Superclass"] != UNMAPPED_MARKER]["ShapeNet_Superclass"].nunique(),
        "unique_objects": metadata_df["Object"].nunique() if "Object" in metadata_df.columns else 0,
        "unique_levels": metadata_df["Level"].nunique() if "Level" in metadata_df.columns else 0,
        "unique_materials": metadata_df["Material"].nunique() if "Material" in metadata_df.columns else 0,
    }
    
    # Class distribution
    shapenet_counts = metadata_df[metadata_df["ShapeNet_Superclass"] != UNMAPPED_MARKER]["ShapeNet_Superclass"].value_counts()
    stats["shapenet_class_distribution"] = shapenet_counts.to_dict()
    
    return stats
