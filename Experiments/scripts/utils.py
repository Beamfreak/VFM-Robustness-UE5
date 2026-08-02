"""
Utility functions for Vision Model Evaluation Framework.
"""

import os
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
from PIL import Image


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic(deterministic: bool = True, benchmark: bool = False):
    """Configure PyTorch determinism settings."""
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    if deterministic:
        # This might raise an error for some operations
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            print("Warning: Could not enable fully deterministic algorithms")


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(filepath: Path) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: Path, indent: int = 2):
    """Save data to JSON file."""
    ensure_dir(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


# ============================================================================
# IMAGE LOADING
# ============================================================================

def load_image(path: str) -> Optional[Image.Image]:
    """
    Safely load an image file.
    
    Args:
        path: Path to image file
        
    Returns:
        PIL Image or None if loading fails
    """
    try:
        img = Image.open(path)
        img = img.convert('RGB')  # Ensure RGB
        return img
    except Exception as e:
        print(f"Warning: Could not load image {path}: {e}")
        return None


def check_image_exists(path: str) -> bool:
    """Check if image file exists and is readable."""
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ============================================================================
# IMAGENET INDEX LOADING
# ============================================================================

def load_imagenet_index(filepath: Path) -> Dict[int, str]:
    """
    Load ImageNet class index file.
    
    Args:
        filepath: Path to imagenet_class_index.txt
        
    Returns:
        Dict mapping index (0-999) to class label string
    """
    import ast
    import re
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to parse as Python dict literal
    try:
        # The file appears to be a Python dict format
        index_map_raw = ast.literal_eval(content)
        # Convert keys to int if needed
        index_map = {int(k): v for k, v in index_map_raw.items()}
        return index_map
    except (ValueError, SyntaxError):
        pass
    
    # Fallback: parse line by line
    index_map = {}
    for line in content.split('\n'):
        line = line.strip().strip('{').strip('}').strip(',')
        if not line:
            continue
        # Match pattern: N: 'label' or N: "label"
        match = re.match(r"(\d+):\s*['\"](.+?)['\"],?$", line)
        if match:
            idx = int(match.group(1))
            label = match.group(2)
            index_map[idx] = label
    
    return index_map


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def compute_top_k_accuracy(y_true: np.ndarray, y_pred_topk: np.ndarray, k: int) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        y_true: True labels (N,)
        y_pred_topk: Top-K predictions (N, K)
        k: K value to use
        
    Returns:
        Top-K accuracy
    """
    if len(y_true) == 0:
        return 0.0
    
    # Check if true label is in top-k predictions
    correct = np.any(y_pred_topk[:, :k] == y_true[:, None], axis=1)
    return float(np.mean(correct))


def bootstrap_ci(
    values: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    statistic: str = "mean"
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval, or normal approximation for large datasets.
    
    Args:
        values: Array of values (e.g., 0/1 for accuracy)
        n_iterations: Number of bootstrap iterations
        confidence: Confidence level (e.g., 0.95 for 95%)
        statistic: "mean" or "median"
        
    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    stat_func = np.mean if statistic == "mean" else np.median
    point_estimate = float(stat_func(values))
    
    # For large datasets, use the normal approximation for the mean to save CPU and memory
    if statistic == "mean" and n > 10000:
        alpha = 1 - confidence
        try:
            import scipy.stats as stats
            z = stats.norm.ppf(1 - alpha / 2)
        except ImportError:
            z_dict = {0.95: 1.95996, 0.99: 2.57583, 0.90: 1.64485}
            z = z_dict.get(confidence, 1.95996)
            
        std_err = float(np.std(values) / np.sqrt(n))
        lower = point_estimate - z * std_err
        upper = point_estimate + z * std_err
        return point_estimate, lower, upper
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_iterations):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return float(point_estimate), float(lower), float(upper)


def optimize_dataframe_types(df: Any) -> Any:
    """
    Optimize memory usage of a pandas DataFrame by downcasting and converting 
    redundant string columns to categorical.
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        return df
        
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            if df[col].empty:
                continue
            max_val = df[col].max()
            min_val = df[col].min()
            if min_val >= 0:
                if max_val < 256:
                    df[col] = df[col].astype("uint8")
                elif max_val < 65536:
                    df[col] = df[col].astype("uint16")
                else:
                    df[col] = df[col].astype("uint32")
            else:
                if min_val >= -128 and max_val <= 127:
                    df[col] = df[col].astype("int8")
                elif min_val >= -32768 and max_val <= 32767:
                    df[col] = df[col].astype("int16")
                else:
                    df[col] = df[col].astype("int32")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_bool_dtype(df[col]):
            pass # Keep as bool
        elif pd.api.types.is_object_dtype(df[col]):
            if col in ["image_path", "Image", "valid_imagenet_indices"]:
                continue
            if "correct" in col.lower():
                df[col] = df[col].astype("boolean")
                continue
            num_unique = df[col].nunique()
            if num_unique < len(df) * 0.5:
                df[col] = df[col].astype("category")
    return df



# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def one_vs_rest_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_class: Any
) -> Tuple[int, int, int, int]:
    """
    Compute TP, TN, FP, FN for one-vs-rest classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_class: The "positive" class
        
    Returns:
        (TP, TN, FP, FN)
    """
    true_positive = y_true == target_class
    pred_positive = y_pred == target_class
    
    tp = int(np.sum(true_positive & pred_positive))
    tn = int(np.sum(~true_positive & ~pred_positive))
    fp = int(np.sum(~true_positive & pred_positive))
    fn = int(np.sum(true_positive & ~pred_positive))
    
    return tp, tn, fp, fn


def compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 from TP, FP, FN.
    
    Returns:
        (precision, recall, f1)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) for top-1 classification predictions.
    """
    if len(confidences) == 0:
        return 0.0
        
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bins, right=True) - 1
    
    ece = 0.0
    n = len(confidences)
    for idx_bin in range(num_bins):
        bin_mask = (bin_indices == idx_bin)
        # Handle predictions exactly at 1.0
        if idx_bin == num_bins - 1:
            bin_mask = bin_mask | (confidences == 1.0)
            
        count = np.sum(bin_mask)
        if count > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (count / n) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute ROC AUC for binary classification or detection tasks.
    """
    if len(np.unique(y_true)) < 2:
        return float('nan')
    
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    if tps[-1] > 0 and fps[-1] > 0:
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]
        auc = np.trapezoid(tpr, fpr)
    else:
        auc = 0.0
        
    return float(auc)


# ============================================================================
# RUN MANIFEST
# ============================================================================

def create_run_manifest(
    config: Any,
    models_evaluated: List[str],
    metadata_path: Path,
    mapping_path: Path,
    code_version: str = "1.0.0"
) -> Dict:
    """
    Create run manifest for reproducibility tracking.
    
    Args:
        config: Configuration object
        models_evaluated: List of model keys evaluated
        metadata_path: Path to metadata CSV
        mapping_path: Path to mapping JSON
        code_version: Version string
        
    Returns:
        Manifest dictionary
    """
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "code_version": code_version,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "metadata_checksum": compute_sha256(metadata_path) if metadata_path.exists() else None,
        "mapping_checksum": compute_sha256(mapping_path) if mapping_path.exists() else None,
        "seed": config.eval.seed if hasattr(config, 'eval') else 42,
        "models_evaluated": models_evaluated,
    }
    return manifest


# ============================================================================
# PROGRESS & LOGGING
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_header(text: str, width: int = 80, char: str = "="):
    """Print formatted header."""
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)
