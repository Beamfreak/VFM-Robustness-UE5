"""
Aggregate comparative results from all datasets into comprehensive tables.

This script:
1. Creates one big table showing model performance across all datasets
2. Generates detailed comparison tables for different metrics
3. Exports to CSV and HTML formats
"""

import json
from pathlib import Path
from typing import Dict
from datetime import datetime

import numpy as np
import pandas as pd

from .utils import ensure_dir


class ComparativeAggregator:
    """Aggregate and analyze comparative results across all datasets."""
    
    # Datasets to process (in order)
    DATASETS = [
        "imagenet-1k", "imagenet-9", "imagenet-a", "imagenet-c", "imagenet-d",
        "imagenet-hard", "imagenet-mini", "imagenet-r", "imagenet-sketch", 
        "imagenet-v2", "tiny-imagenet", "normal_dataset", "plastic_dataset", 
        "objectnet", "PUG_ImageNet"
    ]
    
    # Key metrics to aggregate (ShapeNet removed)
    METRICS = {
        "imagenet_top1": "ImageNet Top-1",
        "imagenet_top5": "ImageNet Top-5",
        "ece": "ECE",
        "roc_auc": "ROC AUC"
    }
    
    # Model variants to include
    MODEL_VARIANTS = ["__knn", "__logits", "__linear_probe"]
    
    def __init__(self, results_dir: Path, output_dir: Path = None):
        """
        Initialize aggregator.
        
        Args:
            results_dir: Path to results directory containing dataset folders
            output_dir: Path to save aggregated tables (default: results_dir/aggregate)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "aggregate"
        
        ensure_dir(self.output_dir)
        
        self.all_data = {}
        self.model_names = {}
        self.filtered_models = {}
    
    def run(self):
        """Execute full aggregation pipeline."""
        print("Aggregating comparative results...")
        
        # Load data from all datasets
        self.load_all_datasets()
        
        # Generate separate aggregations for each eval variant
        results = {}
        
        for variant in ["knn", "logits", "linear_probe"]:
            print(f"\n{'='*60}")
            print(f"Processing {variant.upper()} models")
            print(f"{'='*60}")
            
            # Filter models by variant
            self.filtered_models = {k: v for k, v in self.model_names.items() 
                                    if f"__{variant}" in k}
            
            if not self.filtered_models:
                print(f"⚠ No {variant.upper()} models found")
                continue
            
            # Generate aggregated tables
            master_table = self.create_master_table(variant)
            dataset_specific_tables = self.create_dataset_tables(variant)
            metric_comparison_tables = self.create_metric_comparison_tables(variant)
            model_comparison_tables = self.create_model_comparison_tables(variant)
            
            # Save all tables
            output_subdir = self.output_dir / variant.upper()
            self.save_tables(master_table, dataset_specific_tables, 
                            metric_comparison_tables, model_comparison_tables, 
                            output_subdir)
            
            results[variant] = {
                "master": master_table,
                "by_dataset": dataset_specific_tables,
                "by_metric": metric_comparison_tables,
                "by_model": model_comparison_tables
            }
        
        print(f"\n{'='*60}")
        print("Aggregation complete!")
        print(f"Results saved to:")
        print(f"  - {self.output_dir / 'KNN'}")
        print(f"  - {self.output_dir / 'LOGITS'}")
        print(f"  - {self.output_dir / 'LINEAR_PROBE'}")
        print(f"{'='*60}\n")
        
        return results
    
    def load_all_datasets(self):
        """Load comparative_summary.json from all available datasets."""
        print("\nLoading comparative data from datasets...")
        
        for dataset in self.DATASETS:
            dataset_path = self.results_dir / dataset / "comparative" / "comparative_summary.json"
            
            if not dataset_path.exists():
                print(f"  ⚠ {dataset}: not found")
                continue
            
            try:
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                
                self.all_data[dataset] = data
                
                # Extract model names for consistent naming
                for model_key, model_data in data.get("models", {}).items():
                    self.model_names[model_key] = model_data.get("name", model_key)
                
                print(f"  ✓ {dataset}: {len(data.get('models', {}))} models")
            
            except Exception as e:
                print(f"  ✗ {dataset}: Error loading - {e}")
    
    def create_master_table(self, variant: str = None) -> pd.DataFrame:
        """
        Create master table with model performance across all datasets.
        
        Args:
            variant: "knn" or "logits" to filter models
        
        Returns:
            DataFrame with rows = models, columns = dataset metrics
        """
        print("\nCreating master aggregation table...")
        
        rows = []
        
        # Use filtered models if variant specified
        models_to_process = self.filtered_models if variant else self.model_names
        all_models = sorted(list(models_to_process.keys()))
        
        # Build master table
        for model_key in all_models:
            row = {"Model": self.model_names.get(model_key, model_key)}
            
            top1_values = []
            top5_values = []
            ece_values = []
            roc_values = []
            
            for dataset in self.DATASETS:
                if dataset not in self.all_data:
                    continue
                
                dataset_data = self.all_data[dataset]
                model_data = dataset_data.get("models", {}).get(model_key)
                
                if model_data:
                    img_metrics = model_data.get("imagenet", {})
                    top1 = img_metrics.get("top1_accuracy", np.nan)
                    top5 = img_metrics.get("top5_accuracy", np.nan)
                    ece = img_metrics.get("ece", np.nan)
                    roc = img_metrics.get("roc_auc", np.nan)
                    
                    row[f"{dataset}_Top1"] = top1
                    row[f"{dataset}_Top5"] = top5
                    row[f"{dataset}_ECE"] = ece
                    row[f"{dataset}_ROC"] = roc
                    
                    if not np.isnan(top1):
                        top1_values.append(top1)
                    if not np.isnan(top5):
                        top5_values.append(top5)
                    if not np.isnan(ece):
                        ece_values.append(ece)
                    if not np.isnan(roc):
                        roc_values.append(roc)
                else:
                    for metric_suffix in ["_Top1", "_Top5", "_ECE", "_ROC"]:
                        row[f"{dataset}{metric_suffix}"] = np.nan
            
            # Add averages
            row["Avg_Top1"] = np.mean(top1_values) if top1_values else np.nan
            row["Avg_Top5"] = np.mean(top5_values) if top5_values else np.nan
            row["Avg_ECE"] = np.mean(ece_values) if ece_values else np.nan
            row["Avg_ROC"] = np.mean(roc_values) if roc_values else np.nan
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        print(f"  Master table: {len(df)} models × {len(df.columns)} columns")
        
        return df
    
    def create_dataset_tables(self, variant: str = None) -> Dict[str, pd.DataFrame]:
        """
        Create detailed comparison tables for each dataset.
        
        Args:
            variant: "knn" or "logits" to filter models
        
        Returns:
            Dict mapping dataset name to comparison table
        """
        print("\nCreating per-dataset comparison tables...")
        
        dataset_tables = {}
        models_to_process = self.filtered_models if variant else self.model_names
        
        # First pass: collect all averages for context
        model_averages = {}
        for model_key in models_to_process.keys():
            top1_values = []
            
            for dataset in self.DATASETS:
                if dataset not in self.all_data:
                    continue
                
                dataset_data = self.all_data[dataset]
                model_data = dataset_data.get("models", {}).get(model_key)
                
                if model_data:
                    img_metrics = model_data.get("imagenet", {})
                    top1 = img_metrics.get("top1_accuracy", np.nan)
                    if not np.isnan(top1):
                        top1_values.append(top1)
            
            model_averages[model_key] = np.mean(top1_values) if top1_values else np.nan
        
        for dataset in self.DATASETS:
            if dataset not in self.all_data:
                continue
            
            dataset_data = self.all_data[dataset]
            rows = []
            
            for model_key in sorted(models_to_process.keys()):
                model_data = dataset_data.get("models", {}).get(model_key)
                
                if not model_data:
                    continue
                
                img_metrics = model_data.get("imagenet", {})
                
                row = {
                    "Model": model_data.get("name", model_key),
                    "Model Key": model_key,
                    "ImageNet Top-1": img_metrics.get("top1_accuracy", np.nan),
                    "ImageNet Top-5": img_metrics.get("top5_accuracy", np.nan),
                    "ImageNet ECE": img_metrics.get("ece", np.nan),
                    "ImageNet ROC AUC": img_metrics.get("roc_auc", np.nan),
                    "ImageNet Samples": img_metrics.get("top1_correct", 0),
                    "Avg Top-1 Across All": model_averages.get(model_key, np.nan),
                }
                
                rows.append(row)
            
            if not rows:
                continue
            
            df = pd.DataFrame(rows)
            
            # Sort by ImageNet Top-1 descending
            df = df.sort_values("ImageNet Top-1", ascending=False, na_position="last")
            df.insert(0, "Rank", range(1, len(df) + 1))
            
            dataset_tables[dataset] = df
            print(f"  ✓ {dataset}: {len(df)} models")
        
        return dataset_tables
    
    def create_metric_comparison_tables(self, variant: str = None) -> Dict[str, pd.DataFrame]:
        """
        Create tables comparing specific metrics across datasets.
        
        Args:
            variant: "knn" or "logits" to filter models
        
        Returns:
            Dict mapping metric name to comparison table
        """
        print("\nCreating metric-specific comparison tables...")
        
        metric_tables = {}
        models_to_process = self.filtered_models if variant else self.model_names
        all_models = sorted(list(models_to_process.keys()))
        
        # Create table for each metric (ShapeNet removed)
        metrics_to_extract = [
            ("imagenet.top1_accuracy", "ImageNet Top-1 Accuracy"),
            ("imagenet.top5_accuracy", "ImageNet Top-5 Accuracy"),
            ("imagenet.ece", "ImageNet ECE"),
            ("imagenet.roc_auc", "ImageNet ROC AUC"),
        ]
        
        for metric_path, metric_name in metrics_to_extract:
            parts = metric_path.split(".")
            dataset_key = parts[0]
            metric_key = parts[1]
            
            rows = []
            
            for model_key in all_models:
                row = {"Model": self.model_names.get(model_key, model_key)}
                
                for dataset in self.DATASETS:
                    if dataset not in self.all_data:
                        continue
                    
                    dataset_data = self.all_data[dataset]
                    model_data = dataset_data.get("models", {}).get(model_key)
                    
                    if model_data:
                        ds_metrics = model_data.get(dataset_key, {})
                        value = ds_metrics.get(metric_key, np.nan)
                    else:
                        value = np.nan
                    
                    row[dataset] = value
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Calculate mean and std across datasets
            numeric_cols = [col for col in df.columns if col != "Model"]
            df["Mean"] = df[numeric_cols].mean(axis=1)
            df["Std"] = df[numeric_cols].std(axis=1)
            
            # Sort by mean descending
            df = df.sort_values("Mean", ascending=False, na_position="last")
            
            metric_tables[metric_name] = df
            print(f"  ✓ {metric_name}: {len(df)} models")
        
        return metric_tables
    
    def create_model_comparison_tables(self, variant: str = None) -> Dict[str, pd.DataFrame]:
        """
        Create detailed comparison tables for individual models.
        
        Args:
            variant: "knn" or "logits" to filter models
        
        Returns:
            Dict mapping model key to detailed performance table
        """
        print("\nCreating per-model comparison tables...")
        
        model_tables = {}
        models_to_process = self.filtered_models if variant else self.model_names
        
        for model_key in sorted(models_to_process.keys()):
            rows = []
            top1_vals = []
            top5_vals = []
            ece_vals = []
            roc_vals = []
            
            for dataset in self.DATASETS:
                if dataset not in self.all_data:
                    continue
                
                dataset_data = self.all_data[dataset]
                model_data = dataset_data.get("models", {}).get(model_key)
                
                if model_data:
                    img_metrics = model_data.get("imagenet", {})
                    
                    top1 = img_metrics.get("top1_accuracy", np.nan)
                    top5 = img_metrics.get("top5_accuracy", np.nan)
                    ece = img_metrics.get("ece", np.nan)
                    roc = img_metrics.get("roc_auc", np.nan)
                    
                    row = {
                        "Dataset": dataset,
                        "ImageNet Top-1": top1,
                        "ImageNet Top-5": top5,
                        "ImageNet ECE": ece,
                        "ImageNet ROC AUC": roc,
                    }
                    
                    rows.append(row)
                    
                    if not np.isnan(top1):
                        top1_vals.append(top1)
                    if not np.isnan(top5):
                        top5_vals.append(top5)
                    if not np.isnan(ece):
                        ece_vals.append(ece)
                    if not np.isnan(roc):
                        roc_vals.append(roc)
            
            # Add average row
            if rows:
                avg_row = {
                    "Dataset": "AVERAGE",
                    "ImageNet Top-1": np.mean(top1_vals) if top1_vals else np.nan,
                    "ImageNet Top-5": np.mean(top5_vals) if top5_vals else np.nan,
                    "ImageNet ECE": np.mean(ece_vals) if ece_vals else np.nan,
                    "ImageNet ROC AUC": np.mean(roc_vals) if roc_vals else np.nan,
                }
                rows.append(avg_row)
                
                df = pd.DataFrame(rows)
                model_tables[model_key] = df
        
        print(f"  ✓ Created tables for {len(model_tables)} models")
        
        return model_tables
    
    def save_tables(
        self,
        master_table: pd.DataFrame,
        dataset_tables: Dict[str, pd.DataFrame],
        metric_tables: Dict[str, pd.DataFrame],
        model_tables: Dict[str, pd.DataFrame],
        output_subdir: Path = None
    ):
        """Save all tables to CSV and HTML formats."""
        if output_subdir is None:
            output_subdir = self.output_dir
        
        ensure_dir(output_subdir)
        
        print("\nSaving aggregated tables...")
        
        # Master table
        master_csv = output_subdir / "01_master_all_models_all_datasets.csv"
        master_table.to_csv(master_csv, sep=",", index=False)
        print(f"  ✓ Master table: {master_csv.name}")
        
        # Dataset-specific tables
        dataset_dir = output_subdir / "02_by_dataset"
        ensure_dir(dataset_dir)
        
        for dataset, df in dataset_tables.items():
            csv_path = dataset_dir / f"{dataset}_comparison.csv"
            df.to_csv(csv_path, sep=",", index=False)
        print(f"  ✓ Dataset tables: {len(dataset_tables)} files in {dataset_dir.name}/")
        
        # Metric comparison tables
        metric_dir = output_subdir / "03_by_metric"
        ensure_dir(metric_dir)
        
        for metric_name, df in metric_tables.items():
            clean_name = metric_name.lower().replace(" ", "_").replace(".", "")
            csv_path = metric_dir / f"{clean_name}.csv"
            df.to_csv(csv_path, sep=",", index=False)
        print(f"  ✓ Metric tables: {len(metric_tables)} files in {metric_dir.name}/")
        
        # Model-specific tables
        model_dir = output_subdir / "04_by_model"
        ensure_dir(model_dir)
        
        for model_key, df in model_tables.items():
            clean_name = model_key.lower().replace("_", "-")
            csv_path = model_dir / f"{clean_name}.csv"
            df.to_csv(csv_path, sep=",", index=False)
        print(f"  ✓ Model tables: {len(model_tables)} files in {model_dir.name}/")
        
        # Generate HTML summary
        self._generate_html_summary(master_table, dataset_tables, metric_tables, output_subdir)
    
    def _generate_html_summary(
        self,
        master_table: pd.DataFrame,
        dataset_tables: Dict[str, pd.DataFrame],
        metric_tables: Dict[str, pd.DataFrame],
        output_subdir: Path = None
    ):
        """Generate HTML summary report."""
        if output_subdir is None:
            output_subdir = self.output_dir
        
        html = self._build_html_summary(master_table, dataset_tables, metric_tables)
        
        html_path = output_subdir / "summary.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"  ✓ HTML summary: {html_path.name}")
    
    def _build_html_summary(
        self,
        master_table: pd.DataFrame,
        dataset_tables: Dict[str, pd.DataFrame],
        metric_tables: Dict[str, pd.DataFrame]
    ) -> str:
        """Build HTML summary with embedded tables."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparative Results Aggregation</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a73e8;
            border-bottom: 3px solid #1a73e8;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h2 {{
            color: #202124;
            margin-top: 40px;
            border-left: 4px solid #1a73e8;
            padding-left: 10px;
        }}
        .timestamp {{
            color: #999;
            font-size: 12px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }}
        thead {{
            background: #f8f9fa;
            border-top: 2px solid #dadce0;
            border-bottom: 2px solid #dadce0;
        }}
        th {{
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            color: #202124;
        }}
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #e8eaed;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .numeric {{
            text-align: right;
            font-family: "Courier New", monospace;
            font-size: 12px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: white;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #1a73e8;
        }}
        .stat-label {{
            font-size: 12px;
            color: #5f6368;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #202124;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Comparative Results Aggregation</h1>
        <div class="timestamp">Generated: {now}</div>
        
        <div class="section">
            <h2>Overview</h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">Total Models</div>
                    <div class="stat-value">{len(master_table)}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Datasets Analyzed</div>
                    <div class="stat-value">{len(dataset_tables)}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Metrics Tracked</div>
                    <div class="stat-value">5+</div>
                </div>
            </div>
        </div>
        
        <h2>Master Table (Top 10 Models by Average ImageNet Top-1)</h2>
        <div class="section">
            {self._table_to_html(master_table.head(10))}
        </div>
        
        <h2>Top Performers by Dataset</h2>
        <div class="section">
"""
        
        for dataset, df in sorted(dataset_tables.items())[:5]:  # Top 5 datasets
            top5 = df.head(5)
            html += f"<h3>{dataset}</h3>\n"
            html += self._table_to_html(top5)
        
        html += """
        </div>
        
        <h2>Metric Comparisons</h2>
        <div class="section">
"""
        
        for metric_name, df in sorted(metric_tables.items())[:3]:  # Top 3 metrics
            top5 = df.head(5)
            html += f"<h3>{metric_name}</h3>\n"
            html += self._table_to_html(top5)
        
        html += """
        </div>
        
        <hr style="margin: 40px 0; border: none; border-top: 1px solid #dadce0;">
        <p style="color: #999; font-size: 12px;">
            For detailed comparisons, see CSV files in the aggregate folder.
        </p>
    </div>
</body>
</html>
"""
        return html
    
    def _table_to_html(self, df: pd.DataFrame, max_rows: int = None) -> str:
        """Convert DataFrame to HTML table."""
        if max_rows:
            df = df.head(max_rows)
        
        html = "<table>\n<thead>\n<tr>\n"
        
        for col in df.columns:
            html += f"<th>{col}</th>\n"
        
        html += "</tr>\n</thead>\n<tbody>\n"
        
        for _, row in df.iterrows():
            html += "<tr>\n"
            for col in df.columns:
                value = row[col]
                
                # Format numeric values
                if isinstance(value, (int, float)):
                    if pd.isna(value):
                        formatted = "—"
                    elif col in ["Rank"]:
                        formatted = str(int(value))
                    elif col in ["ImageNet Samples"]:
                        formatted = f"{int(value):,}"
                    else:
                        formatted = f"{value:.4f}"
                    html += f'<td class="numeric">{formatted}</td>\n'
                else:
                    html += f"<td>{value}</td>\n"
            
            html += "</tr>\n"
        
        html += "</tbody>\n</table>\n"
        
        return html


def main():
    """Run comparative results aggregation."""
    results_dir = Path(__file__).parent.parent / "results"
    
    aggregator = ComparativeAggregator(results_dir)
    results = aggregator.run()
    
    return results


if __name__ == "__main__":
    main()
