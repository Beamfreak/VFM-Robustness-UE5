#!/usr/bin/env python3
"""
Fix CSV Path Prefixes in Metadata_Expanded.csv.

Strips redundant dataset folder prefixes from the 'Image' column in `Metadata_Expanded.csv`
files across specified dataset directories.
"""

import argparse
from pathlib import Path
import pandas as pd


def fix_csv_paths(data_dir: Path, dataset_names: list = None):
    """
    Normalizes 'Image' path column across dataset Metadata_Expanded.csv files.
    
    Args:
        data_dir: Root directory containing dataset subfolders.
        dataset_names: Optional list of dataset folder names to process.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    if not dataset_names:
        # Auto-discover all subdirectories containing Metadata_Expanded.csv
        dataset_names = [d.name for d in data_dir.iterdir() if d.is_dir() and (d / "Metadata_Expanded.csv").exists()]

    if not dataset_names:
        print(f"No dataset directories with Metadata_Expanded.csv found in {data_dir}")
        return

    print(f"Processing {len(dataset_names)} datasets in {data_dir}...")

    for ds_name in dataset_names:
        csv_path = data_dir / ds_name / "Metadata_Expanded.csv"
        if not csv_path.exists():
            print(f"Skipping {ds_name}, no Metadata_Expanded.csv found.")
            continue

        try:
            df = pd.read_csv(csv_path, sep=";")
            if "Image" not in df.columns:
                print(f"Skipping {ds_name}, no 'Image' column in CSV.")
                continue

            prefix = f"{ds_name}/"
            folder_prefix = f"{csv_path.parent.name}/"

            def fix_path(p):
                if not isinstance(p, str):
                    return p
                if p.startswith(prefix):
                    return p[len(prefix):]
                if p.startswith(folder_prefix):
                    return p[len(folder_prefix):]
                return p

            old_paths = df["Image"].copy()
            df["Image"] = df["Image"].apply(fix_path)

            changed = (old_paths != df["Image"]).sum()
            if changed > 0:
                df.to_csv(csv_path, sep=";", index=False)
                print(f"  Fixed {changed} paths in {ds_name}")
            else:
                print(f"  No paths needed fixing in {ds_name}")
        except Exception as e:
            print(f"  Error processing {ds_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fix relative path prefixes in Metadata_Expanded.csv files.")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="data",
        help="Root directory containing dataset folders (default: data)",
    )
    parser.add_argument(
        "--datasets",
        "-s",
        nargs="+",
        default=None,
        help="Specific dataset names to process (default: all datasets with Metadata_Expanded.csv)",
    )

    args = parser.parse_args()
    fix_csv_paths(Path(args.data_dir), args.datasets)


if __name__ == "__main__":
    main()
