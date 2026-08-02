"""
Reporting module for Vision Model Evaluation Framework.

Generates HTML, JSON, CSV, and Markdown reports.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd

from .config import MODELS, BASELINE_MODEL
from .utils import save_json, ensure_dir


VARIANT_SEPARATOR = "__"


def _parse_variant_key(model_key: str) -> tuple[str, str]:
    if VARIANT_SEPARATOR in model_key:
        base, mode = model_key.split(VARIANT_SEPARATOR, 1)
        return base, mode
    default_mode = MODELS[model_key].eval_mode if model_key in MODELS else "logits"
    return model_key, default_mode


def _display_model_name(model_key: str) -> str:
    base_key, mode = _parse_variant_key(model_key)
    if base_key in MODELS:
        return f"{MODELS[base_key].name}-{mode.upper()}"
    return model_key


def _model_id_for_key(model_key: str) -> str:
    base_key, _ = _parse_variant_key(model_key)
    if base_key in MODELS:
        return MODELS[base_key].model_id
    return model_key


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """
    Generate evaluation reports in multiple formats.
    """
    
    def __init__(
        self,
        model_key: str,
        metrics: Dict[str, Any],
        predictions_df: pd.DataFrame,
        stratified_analysis: Dict[str, Any],
        output_dir: Path
    ):
        """
        Initialize report generator.
        
        Args:
            model_key: Model identifier
            metrics: Computed metrics dict
            predictions_df: Predictions DataFrame
            stratified_analysis: Stratified analysis results
            output_dir: Output directory for reports
        """
        self.model_key = model_key
        self.model_name = _display_model_name(model_key)
        self.metrics = metrics
        self.predictions_df = predictions_df
        self.stratified = stratified_analysis
        self.output_dir = Path(output_dir)
        
        ensure_dir(self.output_dir)
    
    def generate_all_reports(self) -> Dict[str, Path]:
        """
        Generate all report formats.
        
        Returns:
            Dict mapping format to file path
        """
        paths = {}
        
        paths["json"] = self.generate_json_report()
        paths["csv"] = self.generate_csv_reports()
        paths["html"] = self.generate_html_report()
        paths["markdown"] = self.generate_markdown_report()
        
        return paths
    
    def generate_json_report(self) -> Path:
        """
        Generate JSON metrics summary.
        
        Returns:
            Path to generated file
        """
        filepath = self.output_dir / "metrics_summary.json"
        
        report = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "generated_at": datetime.now().isoformat(),
            "metrics": self.metrics,
            "stratified_analysis": self.stratified
        }
        
        save_json(report, filepath)
        return filepath
    
    def generate_csv_reports(self) -> Dict[str, Path]:
        """
        Generate CSV reports.
        
        Returns:
            Dict mapping report name to file path
        """
        paths = {}
        
        # Predictions
        pred_path = self.output_dir / "predictions.csv"
        self.predictions_df.to_csv(pred_path, sep=";", index=False)
        paths["predictions"] = pred_path
        
        # Per-class ImageNet metrics
        if "per_class_imagenet" in self.metrics and isinstance(self.metrics["per_class_imagenet"], pd.DataFrame):
            pc_img_path = self.output_dir / "metrics_per_class_imagenet.csv"
            self.metrics["per_class_imagenet"].to_csv(pc_img_path, sep=";", index=False)
            paths["per_class_imagenet"] = pc_img_path
        
        # Per-class ShapeNet metrics
        if "per_class_shapenet" in self.metrics and isinstance(self.metrics["per_class_shapenet"], pd.DataFrame):
            pc_sn_path = self.output_dir / "metrics_per_class_shapenet.csv"
            self.metrics["per_class_shapenet"].to_csv(pc_sn_path, sep=";", index=False)
            paths["per_class_shapenet"] = pc_sn_path
        
        # Stratified analysis
        if "per_column" in self.stratified:
            for column, data in self.stratified["per_column"].items():
                clean_name = column.replace(" ", "_").replace("(", "").replace(")", "")
                strat_path = self.output_dir / f"stratified_{clean_name}.csv"
                pd.DataFrame(data).to_csv(strat_path, sep=";", index=False)
                paths[f"stratified_{clean_name}"] = strat_path
        
        return paths
    
    def generate_html_report(self) -> Path:
        """
        Generate HTML report.
        
        Returns:
            Path to generated file
        """
        filepath = self.output_dir / "report.html"
        
        html = self._build_html_report()
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        
        return filepath
    
    def generate_markdown_report(self) -> Path:
        """
        Generate Markdown report.
        
        Returns:
            Path to generated file
        """
        filepath = self.output_dir / "report.md"
        
        md = self._build_markdown_report()
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)
        
        return filepath
    
    def _build_html_report(self) -> str:
        """Build HTML report content."""
        summary = self.metrics.get("summary", {})
        img_metrics = summary.get("imagenet", {})
        sn_metrics = summary.get("shapenet", {})
        
        detailed_img = self.metrics.get("imagenet", {})
        detailed_sn = self.metrics.get("shapenet", {})
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report: {self.model_name}</title>
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
            max-width: 1200px;
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
        }}
        h2 {{
            color: #202124;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #1a73e8;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #5f6368;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #202124;
        }}
        .metric-value.good {{ color: #34a853; }}
        .metric-value.warning {{ color: #fbbc04; }}
        .metric-value.bad {{ color: #ea4335; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e8eaed;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e8eaed;
            color: #5f6368;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Evaluation Report: {self.model_name}</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <h3>Total Samples</h3>
                <div class="metric-value">{summary.get('total_samples', 0):,}</div>
            </div>
            <div class="metric-card">
                <h3>ImageNet Base Top-1</h3>
                <div class="metric-value {self._accuracy_class(img_metrics.get('base_top1_accuracy', 0))}">{img_metrics.get('base_top1_accuracy', 0):.1%}</div>
            </div>
            <div class="metric-card">
                <h3>ImageNet Top-1 Accuracy</h3>
                <div class="metric-value {self._accuracy_class(img_metrics.get('top1_accuracy', 0))}">{img_metrics.get('top1_accuracy', 0):.1%}</div>
            </div>
            <div class="metric-card">
                <h3>ImageNet Top-5 Accuracy</h3>
                <div class="metric-value {self._accuracy_class(img_metrics.get('top5_accuracy', 0))}">{img_metrics.get('top5_accuracy', 0):.1%}</div>
            </div>
            <div class="metric-card">
                <h3>ShapeNet Base Top-1</h3>
                <div class="metric-value {self._accuracy_class(sn_metrics.get('base_top1_accuracy', 0))}">{sn_metrics.get('base_top1_accuracy', 0):.1%}</div>
            </div>
            <div class="metric-card">
                <h3>ShapeNet Top-1 Accuracy</h3>
                <div class="metric-value {self._accuracy_class(sn_metrics.get('top1_accuracy', 0))}">{sn_metrics.get('top1_accuracy', 0):.1%}</div>
            </div>
        </div>
        
        <h2>ImageNet Performance</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Top-1 Accuracy</td><td>{img_metrics.get('top1_accuracy', 0):.4f}</td></tr>
            <tr><td>Top-5 Accuracy</td><td>{img_metrics.get('top5_accuracy', 0):.4f}</td></tr>
            <tr><td>Correct (Top-1)</td><td>{img_metrics.get('top1_correct', 0):,}</td></tr>
            <tr><td>Correct (Top-5)</td><td>{img_metrics.get('top5_correct', 0):,}</td></tr>
            <tr><td>Expected Calibration Error</td><td>{detailed_img.get('expected_calibration_error', float('nan')):.4f}</td></tr>
            <tr><td>ROC AUC</td><td>{detailed_img.get('roc_auc_misclassification', float('nan')):.4f}</td></tr>
        </table>
        
        <h2>ShapeNet Superclass Performance</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Evaluable Samples</td><td>{sn_metrics.get('evaluable_samples', 0):,}</td></tr>
            <tr><td>Coverage</td><td>{sn_metrics.get('coverage_pct', 0):.1f}%</td></tr>
            <tr><td>Top-1 Accuracy</td><td>{sn_metrics.get('top1_accuracy', 0):.4f}</td></tr>
            <tr><td>Correct</td><td>{sn_metrics.get('top1_correct', 0):,}</td></tr>
            <tr><td>Expected Calibration Error</td><td>{detailed_sn.get('expected_calibration_error', float('nan')):.4f}</td></tr>
            <tr><td>ROC AUC</td><td>{detailed_sn.get('roc_auc_misclassification', float('nan')):.4f}</td></tr>
        </table>
        
        {self._build_html_stratified_section()}
        
        <div class="footer">
            <p>Model: {_model_id_for_key(self.model_key)}</p>
            <p>Report generated by Vision Model Evaluation Framework</p>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _build_html_stratified_section(self) -> str:
        """Build HTML for stratified analysis section."""
        if not self.stratified.get("per_column"):
            return ""
        
        sections = ["<h2>Stratified Analysis</h2>"]
        
        for column, data in self.stratified["per_column"].items():
            sections.append(f"<h3>{column}</h3>")
            sections.append("<table>")
            sections.append("<tr><th>Value</th><th>N</th><th>ImageNet Top-1</th><th>ShapeNet Top-1</th></tr>")
            
            for row in data[:10]:  # Show top 10
                sn_acc = row.get('shapenet_top1_acc')
                sn_str = f"{sn_acc:.1%}" if sn_acc is not None else "N/A"
                sections.append(
                    f"<tr><td>{row['value']}</td><td>{row['n_samples']}</td>"
                    f"<td>{row['imagenet_top1_acc']:.1%}</td><td>{sn_str}</td></tr>"
                )
            
            if len(data) > 10:
                sections.append(f"<tr><td colspan='4'>... and {len(data) - 10} more groups</td></tr>")
            
            sections.append("</table>")
        
        return "\n".join(sections)
    
    def _accuracy_class(self, value: float) -> str:
        """Get CSS class based on accuracy value."""
        if value >= 0.7:
            return "good"
        elif value >= 0.4:
            return "warning"
        return "bad"
    
    def _build_markdown_report(self) -> str:
        """Build Markdown report content."""
        summary = self.metrics.get("summary", {})
        img_metrics = summary.get("imagenet", {})
        sn_metrics = summary.get("shapenet", {})
        
        detailed_img = self.metrics.get("imagenet", {})
        detailed_sn = self.metrics.get("shapenet", {})

        md = f"""# Evaluation Report: {self.model_name}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Samples | {summary.get('total_samples', 0):,} |
| ImageNet Base Top-1 Accuracy | {img_metrics.get('base_top1_accuracy', 0):.1%} |
| ImageNet Top-1 Accuracy | {img_metrics.get('top1_accuracy', 0):.1%} |
| ImageNet Top-5 Accuracy | {img_metrics.get('top5_accuracy', 0):.1%} |
| ShapeNet Base Top-1 Accuracy | {sn_metrics.get('base_top1_accuracy', 0):.1%} |
| ShapeNet Top-1 Accuracy | {sn_metrics.get('top1_accuracy', 0):.1%} |

## ImageNet Performance

| Metric | Value |
|--------|-------|
| Base Top-1 Accuracy | {img_metrics.get('base_top1_accuracy', 0):.4f} |
| Base Top-5 Accuracy | {img_metrics.get('base_top5_accuracy', 0):.4f} |
| Top-1 Accuracy | {img_metrics.get('top1_accuracy', 0):.4f} |
| Top-5 Accuracy | {img_metrics.get('top5_accuracy', 0):.4f} |
| Correct (Top-1) | {img_metrics.get('top1_correct', 0):,} |
| Correct (Top-5) | {img_metrics.get('top5_correct', 0):,} |
| Expected Calibration Error | {detailed_img.get('expected_calibration_error', float('nan')):.4f} |
| ROC AUC | {detailed_img.get('roc_auc_misclassification', float('nan')):.4f} |

## ShapeNet Superclass Performance

| Metric | Value |
|--------|-------|
| Evaluable Samples | {sn_metrics.get('evaluable_samples', 0):,} |
| Base Evaluable Samples | {sn_metrics.get('base_evaluable_samples', 0):,} |
| Coverage | {sn_metrics.get('coverage_pct', 0):.1f}% |
| Base Top-1 Accuracy | {sn_metrics.get('base_top1_accuracy', 0):.4f} |
| Top-1 Accuracy | {sn_metrics.get('top1_accuracy', 0):.4f} |
| Correct | {sn_metrics.get('top1_correct', 0):,} |
| Expected Calibration Error | {detailed_sn.get('expected_calibration_error', float('nan')):.4f} |
| ROC AUC | {detailed_sn.get('roc_auc_misclassification', float('nan')):.4f} |
- **Input Size**: {MODELS[_parse_variant_key(self.model_key)[0]].input_size if _parse_variant_key(self.model_key)[0] in MODELS else 'N/A'}

---
*Report generated by Vision Model Evaluation Framework*
"""
        
        return md


# ============================================================================
# COMPARATIVE REPORT
# ============================================================================

class ComparativeReportGenerator:
    """
    Generate cross-model comparative reports.
    """
    
    def __init__(
        self,
        model_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        baseline_key: str = BASELINE_MODEL
    ):
        """
        Initialize comparative report generator.
        
        Args:
            model_results: Dict mapping model_key to results dict
            output_dir: Output directory
            baseline_key: Key of baseline model
        """
        self.model_results = model_results
        self.output_dir = Path(output_dir)
        self.baseline_key = baseline_key
        
        ensure_dir(self.output_dir)
    
    def generate_all_reports(self) -> Dict[str, Path]:
        """Generate all comparative reports."""
        paths = {}
        
        paths["rankings"] = self.generate_rankings_csv()
        paths["rankings_base_models"] = self.generate_rankings_csv(
            filename="model_rankings_base.csv",
            filter_fn=lambda k: ("_b" in k.lower() or "base" in k.lower())
        )
        paths["rankings_small_models"] = self.generate_rankings_csv(
            filename="model_rankings_small.csv",
            filter_fn=lambda k: ("_s" in k.lower() or "small" in k.lower())
        )
        paths["rankings_large_models"] = self.generate_rankings_csv(
            filename="model_rankings_large.csv",
            filter_fn=lambda k: ("_l__" in k.lower() or "_l_" in k.lower() or k.lower().endswith("_l") or "large" in k.lower())
        )
        
        paths["rankings_clip"] = self.generate_rankings_csv("model_rankings_clip.csv", lambda k: "clip" in k.lower())
        paths["rankings_dinov1"] = self.generate_rankings_csv("model_rankings_dinov1.csv", lambda k: "dinov1" in k.lower())
        paths["rankings_dinov2"] = self.generate_rankings_csv("model_rankings_dinov2.csv", lambda k: "dinov2" in k.lower())
        paths["rankings_dinov3"] = self.generate_rankings_csv("model_rankings_dinov3.csv", lambda k: "dinov3" in k.lower())
        paths["rankings_swin"] = self.generate_rankings_csv("model_rankings_swin.csv", lambda k: "swin" in k.lower())
        paths["rankings_hiera"] = self.generate_rankings_csv("model_rankings_hiera.csv", lambda k: "hiera" in k.lower())
        
        paths["imagenet_rankings"] = self.generate_imagenet_rankings_csv()
        paths["json"] = self.generate_json_summary()
        paths["html"] = self.generate_html_report()
        
        return paths
    
    def generate_rankings_csv(self, filename: str = "model_rankings.csv", filter_fn=None) -> Path:
        """Generate CSV with model rankings."""
        rows = []
        
        for model_key, results in self.model_results.items():
            if filter_fn and not filter_fn(model_key):
                continue
            
            summary = results.get("metrics", {}).get("summary", {})
            img = summary.get("imagenet", {})
            sn = summary.get("shapenet", {})
            
            rows.append({
                "model_key": model_key,
                "model_name": _display_model_name(model_key),
                "imagenet_top1": img.get("top1_accuracy", 0),
                "imagenet_top5": img.get("top5_accuracy", 0),
                "shapenet_base_top1": sn.get("base_top1_accuracy", 0),
                "shapenet_top1": sn.get("top1_accuracy", 0),
                "shapenet_evaluable": sn.get("evaluable_samples", 0),
            })
        
        df = pd.DataFrame(rows)
        if df.empty:
            return self.output_dir / filename
            
        df = df.sort_values("shapenet_top1", ascending=False)
        
        # Add rank column
        df.insert(0, "rank", range(1, len(df) + 1))
        
        # Add delta vs baseline
        if self.baseline_key in self.model_results:
            baseline_summary = self.model_results[self.baseline_key].get("metrics", {}).get("summary", {})
            base_sn = baseline_summary.get("shapenet", {}).get("top1_accuracy", 0)
            df["delta_vs_baseline"] = df["shapenet_top1"] - base_sn            
        # Add delta between the model's own Baseperformance and its Average performance
        df["delta_base_to_average"] = df["shapenet_top1"] - df["shapenet_base_top1"]        
        filepath = self.output_dir / filename
        df.to_csv(filepath, sep=";", index=False)
        
        return filepath

    def generate_imagenet_rankings_csv(self) -> Path:
        """Generate CSV with model rankings sorted by ImageNet-1k Top1 Accuracy."""
        rows = []
        
        for model_key, results in self.model_results.items():
            summary = results.get("metrics", {}).get("summary", {})
            img = summary.get("imagenet", {})
            sn = summary.get("shapenet", {})
            
            rows.append({
                "Model": _display_model_name(model_key),
                "IN1k Top1": img.get("top1_accuracy", 0),
                "ShapeNet Top1": sn.get("top1_accuracy", 0),
                "AUROC IN1k": img.get("roc_auc", 0),
                "ECE IN1k": img.get("ece", 0),
            })
            
        df = pd.DataFrame(rows)
        df = df.sort_values("IN1k Top1", ascending=False)
        
        filepath = self.output_dir / "imagenet_rankings.csv"
        df.to_csv(filepath, sep=";", index=False)
        
        return filepath
    
    def generate_json_summary(self) -> Path:
        """Generate JSON summary of all models."""
        summary = {
            "generated_at": datetime.now().isoformat(),
            "baseline_model": self.baseline_key,
            "models": {}
        }
        
        for model_key, results in self.model_results.items():
            model_summary = results.get("metrics", {}).get("summary", {})
            summary["models"][model_key] = {
                "name": _display_model_name(model_key),
                "imagenet": model_summary.get("imagenet", {}),
                "shapenet": model_summary.get("shapenet", {})
            }
        
        filepath = self.output_dir / "comparative_summary.json"
        save_json(summary, filepath)
        
        return filepath
    
    def generate_html_report(self) -> Path:
        """Generate comparative HTML report."""
        rows_data = []
        for model_key, results in self.model_results.items():
            summary = results.get("metrics", {}).get("summary", {})
            img = summary.get("imagenet", {})
            sn = summary.get("shapenet", {})
            
            rows_data.append({
                "key": model_key,
                "name": _display_model_name(model_key),
                "img_top1": img.get("top1_accuracy", 0),
                "img_top5": img.get("top5_accuracy", 0),
                "sn_base_top1": sn.get("base_top1_accuracy", 0),
                "sn_top1": sn.get("top1_accuracy", 0),
            })
        
        # Sort by ShapeNet accuracy
        rows_data.sort(key=lambda x: -x["sn_top1"])
        
        table_rows = []
        for i, row in enumerate(rows_data, 1):
            is_baseline = "⭐" if row["key"] == self.baseline_key else ""
            table_rows.append(
                f"<tr><td>{i}</td><td>{row['name']} {is_baseline}</td>"
                f"<td>{row['img_top1']:.1%}</td><td>{row['img_top5']:.1%}</td>"
                f"<td>{row.get('sn_base_top1', 0):.1%}</td><td>{row['sn_top1']:.1%}</td></tr>"
            )
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Comparative Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 20px; background: #f5f7fa; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e8eaed; }}
        th {{ background: #f8f9fa; }}
        tr:first-child td {{ background: #e8f5e9; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Comparative Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Baseline model: {_display_model_name(self.baseline_key)} ⭐</p>
        
        <h2>Model Rankings (by ShapeNet Top-1 Accuracy)</h2>
        <table>
            <tr><th>Rank</th><th>Model</th><th>ImageNet Top-1</th><th>ImageNet Top-5</th><th>Base ShapeNet Top-1</th><th>ShapeNet Top-1</th></tr>
            {"".join(table_rows)}
        </table>
        
        <div style="margin-top: 40px; color: #5f6368; font-size: 12px;">
            <p>Report generated by Vision Model Evaluation Framework</p>
        </div>
    </div>
</body>
</html>"""
        
        filepath = self.output_dir / "comparative_report.html"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        
        return filepath


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_model_report(
    model_key: str,
    metrics: Dict[str, Any],
    predictions_df: pd.DataFrame,
    stratified_analysis: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Path]:
    """
    Generate all reports for a single model.
    
    Args:
        model_key: Model identifier
        metrics: Computed metrics
        predictions_df: Predictions DataFrame
        stratified_analysis: Stratified analysis results
        output_dir: Output directory
        
    Returns:
        Dict mapping format to file path
    """
    generator = ReportGenerator(
        model_key=model_key,
        metrics=metrics,
        predictions_df=predictions_df,
        stratified_analysis=stratified_analysis,
        output_dir=output_dir
    )
    
    return generator.generate_all_reports()


def generate_comparative_report(
    model_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    baseline_key: str = BASELINE_MODEL
) -> Dict[str, Path]:
    """
    Generate comparative reports across all models.
    
    Args:
        model_results: Dict mapping model_key to results
        output_dir: Output directory
        baseline_key: Baseline model key
        
    Returns:
        Dict mapping format to file path
    """
    generator = ComparativeReportGenerator(
        model_results=model_results,
        output_dir=output_dir,
        baseline_key=baseline_key
    )
    
    return generator.generate_all_reports()
