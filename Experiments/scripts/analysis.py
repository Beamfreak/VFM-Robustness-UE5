"""
Stratified analysis for Vision Model Evaluation Framework.

Computes metrics broken down by metadata attributes.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from .config import STRATIFICATION_COLUMNS, UNMAPPED_MARKER


# ============================================================================
# STRATIFIED ANALYZER
# ============================================================================

class StratifiedAnalyzer:
    """
    Compute metrics stratified by metadata attributes.
    """
    
    def __init__(
        self,
        predictions_df: pd.DataFrame,
        stratify_by: Optional[List[str]] = None
    ):
        """
        Initialize analyzer.
        
        Args:
            predictions_df: DataFrame with predictions
            stratify_by: Columns to stratify by (defaults to STRATIFICATION_COLUMNS)
        """
        self.df = predictions_df
        self.stratify_by = stratify_by or STRATIFICATION_COLUMNS
    
    def analyze_all(self) -> Dict[str, pd.DataFrame]:
        """
        Run stratified analysis on all specified columns.
        
        Returns:
            Dict mapping column name to analysis DataFrame
        """
        results = {}
        
        for column in self.stratify_by:
            if column in self.df.columns:
                results[column] = self.analyze_column(column)
            else:
                print(f"  Warning: Column '{column}' not found in predictions")
        
        return results
    
    def analyze_column(self, column: str) -> pd.DataFrame:
        """
        Compute metrics stratified by a single column.
        
        Args:
            column: Column name to stratify by
            
        Returns:
            DataFrame with metrics per group
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        df_to_analyze = self.df
        
        # Enforce folder-based isolation for Base Variables (Environmental factors)
        # This prevents baseline images (e.g. X=-1.0 Y=0 Z=0.2) from outside folders 
        # from diluting the metrics of the specific characteristic being measured.
        folder_mapping = {
            "Camera Position": "Camera",
            "Fog": "Fog",
            "Light Color (RGB)": "Light",
            "Material": "Material",
            "Background": "Background",
            "Level": "Background"
        }
        
        if column in folder_mapping and "image_path" in self.df.columns:
            target_folder = folder_mapping[column]
            # Match either the specific target folder OR the baseline folder (Camera/Angle0)
            # This ensures we keep the 'control' group (e.g. Fog=False, Level=BaseMap) 
            # without diluting it with other variations.
            target_mask = self.df["image_path"].str.contains(rf'(?:^|/){target_folder}/', regex=True, na=False)
            baseline_mask = self.df["image_path"].str.contains(r'(?:^|/)Camera/Angle0/', regex=True, na=False)
            
            df_to_analyze = self.df[target_mask | baseline_mask].copy()
            df_to_analyze[column] = df_to_analyze[column].astype(str)
            
            # Fix incorrect Light Color defaults in metadata
            if column == "Light Color (RGB)":
                is_not_light_folder = ~df_to_analyze["image_path"].str.contains(rf'(?:^|/){target_folder}/', regex=True, na=False)
                df_to_analyze.loc[is_not_light_folder, column] = "(R=1.000000,G=1.000000,B=1.000000,A=0.000000)"
        
        groups = df_to_analyze.groupby(column)
        
        rows = []
        for group_value, group_df in groups:
            n = len(group_df)
            
            # ImageNet metrics
            img_top1_acc = group_df["imagenet_top1_correct"].mean()
            img_top5_acc = group_df["imagenet_top5_correct"].mean()
            
            # ShapeNet metrics (evaluable only)
            sn_mask = group_df["shapenet_evaluable"].copy()
            sn_n = sn_mask.sum()
            sn_acc = group_df.loc[sn_mask, "shapenet_top1_correct"].fillna(False).mean() if sn_n > 0 else None
            
            rows.append({
                "value": group_value,
                "n_samples": n,
                "imagenet_top1_acc": float(img_top1_acc),
                "imagenet_top5_acc": float(img_top5_acc),
                "shapenet_evaluable": int(sn_n),
                "shapenet_top1_acc": float(sn_acc) if sn_acc is not None else None,
            })
        
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("n_samples", ascending=False)
        
        return result
    
    def compute_variance_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Compute variance of metrics across stratification groups.
        
        Returns:
            Dict with variance statistics per column
        """
        results = {}
        
        for column in self.stratify_by:
            if column not in self.df.columns:
                continue
            
            analysis = self.analyze_column(column)
            if analysis.empty:
                continue
            
            # Compute variance of accuracies
            img_top1_values = analysis["imagenet_top1_acc"].dropna().values
            sn_values = analysis["shapenet_top1_acc"].dropna().values
            
            results[column] = {
                "n_groups": len(analysis),
                "imagenet_top1_variance": float(np.var(img_top1_values)) if len(img_top1_values) > 0 else None,
                "imagenet_top1_std": float(np.std(img_top1_values)) if len(img_top1_values) > 0 else None,
                "imagenet_top1_range": float(np.ptp(img_top1_values)) if len(img_top1_values) > 0 else None,
                "shapenet_top1_variance": float(np.var(sn_values)) if len(sn_values) > 0 else None,
                "shapenet_top1_std": float(np.std(sn_values)) if len(sn_values) > 0 else None,
                "shapenet_top1_range": float(np.ptp(sn_values)) if len(sn_values) > 0 else None,
            }
        
        return results
    
    def find_best_worst_subgroups(
        self,
        metric: str = "shapenet_top1_acc",
        top_n: int = 5
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Find best and worst performing subgroups.
        
        Args:
            metric: Metric to rank by
            top_n: Number of top/bottom groups to return
            
        Returns:
            Dict with best/worst subgroups per column
        """
        results = {}
        
        for column in self.stratify_by:
            if column not in self.df.columns:
                continue
            
            analysis = self.analyze_column(column)
            if analysis.empty or metric not in analysis.columns:
                continue
            
            # Filter out None values
            valid = analysis[analysis[metric].notna()].copy()
            if len(valid) == 0:
                continue
            
            # Sort by metric
            sorted_df = valid.sort_values(metric, ascending=False)
            
            best = sorted_df.head(top_n).to_dict("records")
            worst = sorted_df.tail(top_n).to_dict("records")
            
            results[column] = {
                "best": best,
                "worst": worst
            }
        
        return results


# ============================================================================
# CROSS-MODEL COMPARISON
# ============================================================================

def compare_models_stratified(
    model_predictions: Dict[str, pd.DataFrame],
    column: str
) -> pd.DataFrame:
    """
    Compare multiple models' performance on a stratified column.
    
    Args:
        model_predictions: Dict mapping model_key to predictions DataFrame
        column: Column to stratify by
        
    Returns:
        DataFrame with all models' metrics per group
    """
    all_rows = []
    
    for model_key, preds_df in model_predictions.items():
        if column not in preds_df.columns:
            continue
        
        analyzer = StratifiedAnalyzer(preds_df, stratify_by=[column])
        analysis = analyzer.analyze_column(column)
        
        for _, row in analysis.iterrows():
            all_rows.append({
                "model": model_key,
                "group": row["value"],
                "n_samples": row["n_samples"],
                "imagenet_top1": row["imagenet_top1_acc"],
                "shapenet_top1": row["shapenet_top1_acc"],
            })
    
    result = pd.DataFrame(all_rows)
    
    # Pivot for easier comparison
    if len(result) > 0:
        pivot = result.pivot_table(
            index="group",
            columns="model",
            values=["imagenet_top1", "shapenet_top1"],
            aggfunc="first"
        )
        return pivot
    
    return result


def find_model_strengths(
    model_predictions: Dict[str, pd.DataFrame],
    stratify_by: Optional[List[str]] = None,
    metric: str = "shapenet_top1_acc"
) -> Dict[str, Dict[str, List[str]]]:
    """
    Find which model performs best on each subgroup.
    
    Args:
        model_predictions: Dict mapping model_key to predictions DataFrame
        stratify_by: Columns to analyze
        metric: Metric to compare
        
    Returns:
        Dict mapping column to dict of (group_value -> winning models)
    """
    stratify_by = stratify_by or STRATIFICATION_COLUMNS
    results = {}
    
    for column in stratify_by:
        column_results = {}
        
        # Get all model scores for each group
        group_scores = {}  # group_value -> {model: score}
        
        for model_key, preds_df in model_predictions.items():
            if column not in preds_df.columns:
                continue
            
            analyzer = StratifiedAnalyzer(preds_df, stratify_by=[column])
            analysis = analyzer.analyze_column(column)
            
            for _, row in analysis.iterrows():
                group_val = row["value"]
                score = row[metric]
                
                if score is None or np.isnan(score):
                    continue
                
                if group_val not in group_scores:
                    group_scores[group_val] = {}
                group_scores[group_val][model_key] = score
        
        # Find winner for each group
        for group_val, scores in group_scores.items():
            if not scores:
                continue
            max_score = max(scores.values())
            winners = [m for m, s in scores.items() if s == max_score]
            column_results[group_val] = winners
        
        results[column] = column_results
    
    return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_stratified_analysis(
    predictions_df: pd.DataFrame,
    stratify_by: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run full stratified analysis.
    
    Args:
        predictions_df: Predictions DataFrame
        stratify_by: Columns to stratify by
        
    Returns:
        Dict with all analysis results
    """
    analyzer = StratifiedAnalyzer(predictions_df, stratify_by)
    
    results = {
        "per_column": {},
        "variance": analyzer.compute_variance_analysis(),
        "best_worst": analyzer.find_best_worst_subgroups()
    }
    
    # Per-column analysis
    for column_analysis in analyzer.analyze_all().items():
        column, df = column_analysis
        results["per_column"][column] = df.to_dict("records")
    
    return results


def export_stratified_analysis(
    analysis_results: Dict[str, pd.DataFrame],
    output_dir: str,
    prefix: str = ""
):
    """
    Export stratified analysis to CSV files.
    
    Args:
        analysis_results: Dict from analyze_all()
        output_dir: Directory to save files
        prefix: Optional filename prefix
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for column, df in analysis_results.items():
        # Clean filename
        clean_name = column.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"{prefix}stratified_{clean_name}.csv" if prefix else f"stratified_{clean_name}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False, sep=";")
