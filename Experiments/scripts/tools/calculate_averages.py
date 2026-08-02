#!/usr/bin/env python3
"""
Calculate Average Top-1 Accuracies Across Stratified Factors.

Processes `stratified_*.csv` files in model result directories for a given dataset,
filtering out 'Default' values and computing per-variable average accuracies per model.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd


def calculate_averages(dataset_results_dir: Path, output_dir: Path = None):
    """
    Calculate factor averages for all model runs in dataset_results_dir.
    
    Args:
        dataset_results_dir: Path to dataset results folder (e.g. results/PUG_ImageNet).
        output_dir: Optional custom output directory. Defaults to dataset_results_dir.
    """
    dataset_results_dir = Path(dataset_results_dir)
    if not dataset_results_dir.exists():
        print(f"Error: Directory '{dataset_results_dir}' does not exist.")
        sys.exit(1)

    output_dir = Path(output_dir) if output_dir else dataset_results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary: {variable_name: {model_name: avg_top1_acc}}
    results = defaultdict(dict)

    # Find model directories
    model_dirs = sorted([d for d in dataset_results_dir.iterdir() if d.is_dir() and d.name != "comparative"])

    if not model_dirs:
        print(f"No model directories found in {dataset_results_dir}")
        return

    print(f"Found {len(model_dirs)} model directories in {dataset_results_dir}\n")

    for model_dir in model_dirs:
        model_name = model_dir.name
        csv_files = sorted(model_dir.glob("stratified_*.csv"))

        for csv_file in csv_files:
            var_name = csv_file.stem.replace("stratified_", "")
            try:
                df = pd.read_csv(csv_file, sep=";")
                if "value" not in df.columns:
                    continue

                df_filtered = df[df["value"] != "Default"]

                # Check available accuracy column
                acc_col = None
                for col in ["imagenet_top1_acc", "top1_acc", "accuracy"]:
                    if col in df_filtered.columns:
                        acc_col = col
                        break

                if acc_col and len(df_filtered) > 0:
                    avg_top1 = df_filtered[acc_col].mean()
                    results[var_name][model_name] = avg_top1
            except Exception as e:
                print(f"  Error processing {csv_file.name} in {model_name}: {e}")

    if not results:
        print("No valid stratified data found.")
        return

    # Convert results to DataFrame
    output_data = {var_name: pd.Series(results[var_name]) for var_name in sorted(results.keys())}
    output_df = pd.DataFrame(output_data).sort_index()

    print("\n" + "=" * 80)
    print("AVERAGE TOP1-ACCURACIES BY VARIABLE (excluding Default)")
    print("=" * 80 + "\n")
    print(output_df.to_string())

    # Save outputs
    output_csv = output_dir / "averaged_top1_accuracies.csv"
    output_df.to_csv(output_csv)
    print(f"\nResults saved to: {output_csv}")

    # Summary statistics across models
    summary_stats = output_df.describe().T[["mean", "std", "min", "max"]]
    summary_csv = output_dir / "averaged_top1_accuracies_summary.csv"
    summary_stats.to_csv(summary_csv)
    print(f"Summary stats saved to: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description="Calculate average accuracies across factor levels.")
    parser.add_argument(
        "--dataset-results-dir",
        "-d",
        type=str,
        default="results/PUG_ImageNet",
        help="Path to dataset results folder (default: results/PUG_ImageNet)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save summary CSVs (default: same as dataset-results-dir)",
    )

    args = parser.parse_args()
    calculate_averages(Path(args.dataset_results_dir), Path(args.output_dir) if args.output_dir else None)


if __name__ == "__main__":
    main()
