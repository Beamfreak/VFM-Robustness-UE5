"""
Metrics computation for Vision Model Evaluation Framework.

Computes ImageNet and ShapeNet classification metrics.
"""

from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict

from .config import UNMAPPED_MARKER, DATASETS, DEFAULT_DATASET
from .utils import (
    compute_accuracy,
    compute_top_k_accuracy,
    bootstrap_ci,
    one_vs_rest_metrics,
    compute_prf,
    compute_ece,
    compute_roc_auc
)


# ============================================================================
# METRICS COMPUTER
# ============================================================================

class MetricsComputer:
    """
    Compute comprehensive evaluation metrics from predictions.
    """
    
    def __init__(
        self,
        predictions_df: pd.DataFrame,
        bootstrap_iterations: int = 1000,
        confidence_level: float = 0.95,
        dataset_key: str = DEFAULT_DATASET
    ):
        """
        Initialize metrics computer.
        
        Args:
            predictions_df: DataFrame with predictions (from inference)
            bootstrap_iterations: Number of bootstrap iterations for CI
            confidence_level: Confidence level for intervals
            dataset_key: Key for the dataset specification to use
        """
        self.df = predictions_df
        self.bootstrap_iterations = bootstrap_iterations
        self.confidence_level = confidence_level
        self.dataset_key = dataset_key
        self.dataset_spec = DATASETS[self.dataset_key]
    
    def _get_base_mask(self, df_subset: pd.DataFrame) -> pd.Series:
        """Helper to get mask for baseline images."""
        if "image_path" not in df_subset.columns or not self.dataset_spec.baseline_dir:
            return pd.Series(False, index=df_subset.index)
        # Using a simple substring match for the baseline_dir
        # Since baseline_dir might have special regex characters, we use regex=False or escape it
        return df_subset["image_path"].str.contains(
            f"/{self.dataset_spec.baseline_dir}|^" + self.dataset_spec.baseline_dir, 
            case=False, na=False, regex=True
        )
    
    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        Compute all metrics.
        
        Returns:
            Dict with all computed metrics
        """
        results = {
            "summary": self.compute_summary_metrics(),
            "imagenet": self.compute_imagenet_metrics(),
            "shapenet": self.compute_shapenet_metrics(),
            "per_class_imagenet": self.compute_per_class_imagenet_metrics(),
            "per_class_shapenet": self.compute_per_class_shapenet_metrics(),
        }
        
        return results
    
    def compute_summary_metrics(self) -> Dict[str, Any]:
        """
        Compute high-level summary metrics.
        
        Returns:
            Dict with summary statistics
        """
        total = len(self.df)
        
        # ImageNet metrics
        imagenet_top1_correct = self.df["imagenet_top1_correct"].sum()
        imagenet_top5_correct = self.df["imagenet_top5_correct"].sum()
        
        # Base Image Definition
        is_base = self._get_base_mask(self.df)
            
        base_samples_total = is_base.sum()
        base_imagenet_top1 = self.df.loc[is_base, "imagenet_top1_correct"].sum() if base_samples_total > 0 else 0
        base_imagenet_top5 = self.df.loc[is_base, "imagenet_top5_correct"].sum() if base_samples_total > 0 else 0
        
        # Calculate ImageNet ECE and ROC AUC
        if "top1_prob" in self.df.columns:
            confidences = self.df["top1_prob"].to_numpy().astype(float)
            img_top1_arr = self.df["imagenet_top1_correct"].astype(float).to_numpy()
            img_ece = compute_ece(confidences, img_top1_arr)
            img_roc_auc = compute_roc_auc(img_top1_arr.astype(int), confidences)
        else:
            img_ece = float('nan')
            img_roc_auc = float('nan')
        
        # ShapeNet metrics (only evaluable samples based on ground truth)
        shapenet_mask = self.df["shapenet_evaluable"].copy()
        
        shapenet_evaluable = shapenet_mask.sum()
        shapenet_correct_arr = self.df.loc[shapenet_mask, "shapenet_top1_correct"].fillna(False)
        shapenet_correct = shapenet_correct_arr.sum()
        
        is_base_shapenet = is_base & shapenet_mask
        base_sn_samples = is_base_shapenet.sum()
        base_sn_top1 = self.df.loc[is_base_shapenet, "shapenet_top1_correct"].fillna(False).sum() if base_sn_samples > 0 else 0
        
        # Calculate ShapeNet ECE and ROC AUC
        if "top1_prob" in self.df.columns and shapenet_evaluable > 0:
            sn_confidences = self.df.loc[shapenet_mask, "top1_prob"].to_numpy().astype(float)
            sn_correct_arr_float = shapenet_correct_arr.astype(float).to_numpy()
            sn_ece = compute_ece(sn_confidences, sn_correct_arr_float)
            sn_roc_auc = compute_roc_auc(sn_correct_arr_float.astype(int), sn_confidences)
        else:
            sn_ece = float('nan')
            sn_roc_auc = float('nan')
        
        summary = {
            "total_samples": total,
            "base_samples": int(base_samples_total),
            "imagenet": {
                "top1_accuracy": imagenet_top1_correct / total if total > 0 else 0,
                "top5_accuracy": imagenet_top5_correct / total if total > 0 else 0,
                "base_top1_accuracy": float(base_imagenet_top1 / base_samples_total) if base_samples_total > 0 else 0.0,
                "base_top5_accuracy": float(base_imagenet_top5 / base_samples_total) if base_samples_total > 0 else 0.0,
                "top1_correct": int(imagenet_top1_correct),
                "top5_correct": int(imagenet_top5_correct),
                "ece": float(img_ece),
                "roc_auc": float(img_roc_auc),
            },
            "shapenet": {
                "evaluable_samples": int(shapenet_evaluable),
                "base_evaluable_samples": int(base_sn_samples),
                "coverage_pct": 100 * shapenet_evaluable / total if total > 0 else 0,
                "top1_accuracy": shapenet_correct / shapenet_evaluable if shapenet_evaluable > 0 else 0,
                "base_top1_accuracy": float(base_sn_top1 / base_sn_samples) if base_sn_samples > 0 else 0.0,
                "top1_correct": int(shapenet_correct),
                "ece": float(sn_ece),
                "roc_auc": float(sn_roc_auc),
            }
        }
        
        return summary
    
    def compute_imagenet_metrics(self) -> Dict[str, Any]:
        """
        Compute detailed ImageNet classification metrics.
        
        Returns:
            Dict with ImageNet metrics
        """
        n = len(self.df)
        
        # Basic accuracy
        top1_correct = self.df["imagenet_top1_correct"].to_numpy()
        top5_correct = self.df["imagenet_top5_correct"].to_numpy()
        
        top1_acc = np.mean(top1_correct)
        top5_acc = np.mean(top5_correct)
        
        # Bootstrap CI
        top1_ci = bootstrap_ci(
            top1_correct.astype(float),
            self.bootstrap_iterations,
            self.confidence_level
        )
        top5_ci = bootstrap_ci(
            top5_correct.astype(float),
            self.bootstrap_iterations,
            self.confidence_level
        )
        
        # Calibration and Detection Metrics
        if "top1_prob" in self.df.columns:
            confidences = self.df["top1_prob"].to_numpy().astype(float)
            ece = compute_ece(confidences, top1_correct.astype(float))
            roc_auc = compute_roc_auc(top1_correct.astype(int), confidences)
        else:
            ece = float('nan')
            roc_auc = float('nan')
        
        metrics = {
            "n_samples": n,
            "top1_accuracy": float(top1_acc),
            "top1_accuracy_ci": {
                "lower": top1_ci[1],
                "upper": top1_ci[2],
                "confidence": self.confidence_level
            },
            "top5_accuracy": float(top5_acc),
            "top5_accuracy_ci": {
                "lower": top5_ci[1],
                "upper": top5_ci[2],
                "confidence": self.confidence_level
            },
            "top1_error_rate": 1.0 - float(top1_acc),
            "top5_error_rate": 1.0 - float(top5_acc),
            "expected_calibration_error": ece,
            "roc_auc_misclassification": roc_auc,
        }
        
        # Confusion analysis (top confused pairs)
        metrics["top_confused_pairs"] = self._get_top_confused_pairs(
            self.df["true_imagenet_idx"].to_numpy(),
            self.df["top1_idx"].to_numpy(),
            top_n=10
        )
        
        return metrics
    
    def compute_shapenet_metrics(self) -> Dict[str, Any]:
        """
        Compute detailed ShapeNet superclass metrics.
        
        Returns:
            Dict with ShapeNet metrics
        """
# Filter to evaluable samples (based only on ground truth)
        mask = self.df["shapenet_evaluable"].copy()
        
        eval_df = self.df[mask]
        n = len(eval_df)

        if n == 0:
            return {
                "n_samples": 0,
                "top1_accuracy": 0.0,
                "message": "No evaluable ShapeNet samples"
            }

        # Basic accuracy - fillna(False) treats unmapped predictions as incorrect
        correct = eval_df["shapenet_top1_correct"].fillna(False).to_numpy().astype(float)
        accuracy = np.mean(correct)
        
        # Bootstrap CI
        acc_ci = bootstrap_ci(
            correct,
            self.bootstrap_iterations,
            self.confidence_level
        )
        
        # Calibration and Detection Metrics
        if "top1_prob" in eval_df.columns:
            confidences = eval_df["top1_prob"].to_numpy().astype(float)
            ece = compute_ece(confidences, correct)
            roc_auc = compute_roc_auc(correct.astype(int), confidences)
        else:
            ece = float('nan')
            roc_auc = float('nan')
        
        # Unique classes
        true_classes = eval_df["true_shapenet_superclass"].unique().tolist()
        pred_classes = eval_df["pred_shapenet_top1"].unique().tolist()
        
        metrics = {
            "n_samples": n,
            "n_excluded": len(self.df) - n,
            "top1_accuracy": float(accuracy),
            "top1_accuracy_ci": {
                "lower": acc_ci[1],
                "upper": acc_ci[2],
                "confidence": self.confidence_level
            },
            "error_rate": 1.0 - float(accuracy),
            "expected_calibration_error": ece,
            "roc_auc_misclassification": roc_auc,
            "n_ground_truth_classes": len(true_classes),
            "n_predicted_classes": len(pred_classes),
            "ground_truth_classes": sorted(true_classes),
        }
        
        # Confusion analysis
        metrics["top_confused_pairs"] = self._get_top_confused_pairs(
            eval_df["true_shapenet_superclass"].to_numpy(),
            eval_df["pred_shapenet_top1"].to_numpy(),
            top_n=10
        )
        
        return metrics
    
    def compute_per_class_imagenet_metrics(self) -> pd.DataFrame:
        """
        Compute per-class metrics for ImageNet.
        
        Returns:
            DataFrame with per-class precision, recall, F1
        """
        y_true = self.df["true_imagenet_idx"].to_numpy()
        y_pred = self.df["top1_idx"].to_numpy()
        
        # Identify base samples
        is_base = self._get_base_mask(self.df)
            
        base_correct = self.df["imagenet_top1_correct"]
        
        # Get unique classes
        classes = np.unique(y_true)
        
        rows = []
        for cls in classes:
            class_mask = (y_true == cls)
            cls_base_mask = is_base & class_mask
            cls_base_n = cls_base_mask.sum()
            cls_base_acc = base_correct[cls_base_mask].mean() if cls_base_n > 0 else float('nan')
            
            cls_overall_acc = base_correct[class_mask].mean() if class_mask.sum() > 0 else 0.0
            
            tp, tn, fp, fn = one_vs_rest_metrics(y_true, y_pred, cls)
            precision, recall, f1 = compute_prf(tp, fp, fn)
            
            # Get label
            label_matches = self.df[self.df["true_imagenet_idx"] == cls]
            label = label_matches["true_imagenet_label"].iloc[0] if len(label_matches) > 0 else ""
            
            rows.append({
                "class_idx": int(cls),
                "class_label": label,
                "support": int(tp + fn),
                "base_accuracy": float(cls_base_acc),
                "overall_accuracy": float(cls_overall_acc),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values("f1", ascending=False)
        
        return df
    
    def compute_per_class_shapenet_metrics(self) -> pd.DataFrame:
        """
        Compute per-class metrics for ShapeNet superclasses.
        
        Returns:
            DataFrame with per-superclass precision, recall, F1
        """
# Filter to evaluable (based only on ground truth)
        mask = self.df["shapenet_evaluable"].copy()
        
        eval_df = self.df[mask]
        
        if len(eval_df) == 0:
            return pd.DataFrame()
        
        y_true = eval_df["true_shapenet_superclass"].to_numpy()
        y_pred = eval_df["pred_shapenet_top1"].to_numpy()
        
        # Identify base samples within evaluable df
        is_base = self._get_base_mask(eval_df)
            
        base_correct = eval_df["shapenet_top1_correct"].fillna(False)
        
        # Get unique classes
        classes = np.unique(y_true)
        
        rows = []
        for cls in classes:
            class_mask = (y_true == cls)
            cls_base_mask = is_base & class_mask
            cls_base_n = cls_base_mask.sum()
            cls_base_acc = base_correct[cls_base_mask].mean() if cls_base_n > 0 else float('nan')
            
            # Accuracy for this class overall
            class_acc = base_correct[class_mask].mean() if class_mask.sum() > 0 else 0.0
            
            tp, tn, fp, fn = one_vs_rest_metrics(y_true, y_pred, cls)
            precision, recall, f1 = compute_prf(tp, fp, fn)
            
            rows.append({
                "superclass": cls,
                "support": int(tp + fn),
                "base_accuracy": float(cls_base_acc),
                "overall_accuracy": float(class_acc),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values("f1", ascending=False)
        
        return df
    
    def compute_confusion_matrix(
        self,
        metric_type: str = "shapenet"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute confusion matrix.
        
        Args:
            metric_type: "imagenet" or "shapenet"
            
        Returns:
            (confusion_matrix, labels)
        """
        if metric_type == "shapenet":
            mask = self.df["shapenet_evaluable"].copy()
            y_true = self.df.loc[mask, "true_shapenet_superclass"].to_numpy()
            y_pred = self.df.loc[mask, "pred_shapenet_top1"].to_numpy()
        else:
            y_true = self.df["true_imagenet_idx"].to_numpy()
            y_pred = self.df["top1_idx"].to_numpy()
        
        # Get unique labels
        labels = sorted(set(y_true) | set(y_pred))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        
        # Build matrix
        n = len(labels)
        matrix = np.zeros((n, n), dtype=int)
        
        for t, p in zip(y_true, y_pred):
            i = label_to_idx[t]
            j = label_to_idx[p]
            matrix[i, j] += 1
        
        return matrix, labels
    
    def _get_top_confused_pairs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_n: int = 10
    ) -> List[Dict]:
        """
        Get top confused class pairs.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            top_n: Number of top pairs to return
            
        Returns:
            List of dicts with (true, pred, count)
        """
        # Only look at incorrect predictions
        incorrect_mask = y_true != y_pred
        true_incorrect = y_true[incorrect_mask]
        pred_incorrect = y_pred[incorrect_mask]
        
        # Count pairs
        pair_counts = defaultdict(int)
        for t, p in zip(true_incorrect, pred_incorrect):
            pair_counts[(t, p)] += 1
        
        # Sort by count
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
        
        result = []
        for (true_cls, pred_cls), count in sorted_pairs[:top_n]:
            result.append({
                "true_class": true_cls if isinstance(true_cls, str) else int(true_cls),
                "predicted_class": pred_cls if isinstance(pred_cls, str) else int(pred_cls),
                "count": count
            })
        
        return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def compute_metrics_from_predictions(
    predictions_df: pd.DataFrame,
    bootstrap_iterations: int = 1000,
    confidence_level: float = 0.95,
    dataset_key: str = DEFAULT_DATASET
) -> Dict[str, Any]:
    """
    Compute all metrics from predictions DataFrame.
    
    Args:
        predictions_df: DataFrame with predictions
        bootstrap_iterations: Bootstrap iterations for CI
        confidence_level: Confidence level
        dataset_key: Key for the dataset specification to use
        
    Returns:
        Dict with all metrics
    """
    computer = MetricsComputer(
        predictions_df=predictions_df,
        bootstrap_iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        dataset_key=dataset_key
    )
    return computer.compute_all_metrics()


def compute_delta_metrics(
    model_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute delta between model and baseline.
    
    Args:
        model_metrics: Metrics from target model
        baseline_metrics: Metrics from baseline model
        
    Returns:
        Dict with delta values
    """
    deltas = {}
    
    # ImageNet deltas
    model_img = model_metrics.get("summary", {}).get("imagenet", {})
    base_img = baseline_metrics.get("summary", {}).get("imagenet", {})
    
    deltas["imagenet"] = {
        "top1_accuracy_delta": model_img.get("top1_accuracy", 0) - base_img.get("top1_accuracy", 0),
        "top5_accuracy_delta": model_img.get("top5_accuracy", 0) - base_img.get("top5_accuracy", 0),
    }
    
    # ShapeNet deltas
    model_sn = model_metrics.get("summary", {}).get("shapenet", {})
    base_sn = baseline_metrics.get("summary", {}).get("shapenet", {})
    
    deltas["shapenet"] = {
        "top1_accuracy_delta": model_sn.get("top1_accuracy", 0) - base_sn.get("top1_accuracy", 0),
    }
    
    return deltas
