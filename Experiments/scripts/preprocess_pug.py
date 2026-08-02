"""
PUG-ImageNet Metadata Preprocessor

Converts the PUG-ImageNet labels CSV into the standard Metadata_Expanded.csv
format expected by the evaluation framework.

Usage:
    python preprocess_pug.py
    python preprocess_pug.py --data-dir /path/to/pug_imagenet --output /path/to/out.csv
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def preprocess_pug(source_csv: Path, mapping_json: Path, output_csv: Path) -> None:
    df = pd.read_csv(source_csv)

    with open(mapping_json, "r") as f:
        class_mapping = json.load(f)

    metadata = []
    unmapped_count = 0

    for _, row in df.iterrows():
        char_label = row["character_label"]

        if char_label in class_mapping and len(class_mapping[char_label]) > 0:
            # When multiple ImageNet indices match, keep the first for single-label eval
            imagenet_idx = class_mapping[char_label][0]
            valid_classes = ",".join(map(str, class_mapping[char_label]))
        else:
            imagenet_idx = -1
            valid_classes = ""
            unmapped_count += 1

        entry = {
            "Image": row["filename"],
            "Object": row["character_name"],
            "Level": "<unmapped>",
            "Class": imagenet_idx,
            "Valid_Classes": valid_classes,
            "Material": row["character_texture"],
            "Camera Position": f"Pitch:{row['camera_pitch']}_Yaw:{row['camera_yaw']}_Roll:{row['camera_roll']}",
            "Light Color (RGB)": row["scene_light"],
            "Fog": "<unmapped>",
            "ImageNet_Label": char_label,
            "ShapeNet_Superclass": "<unmapped>",
        }
        metadata.append(entry)

    out_df = pd.DataFrame(metadata)

    # Drop rows that could not be mapped to a valid ImageNet index
    out_df = out_df[out_df["Class"] != -1]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, sep=";")

    print(f"Conversion complete.")
    print(f"Written to: {output_csv}")
    print(f"Total rows exported: {len(out_df)}")
    if unmapped_count > 0:
        print(f"Skipped {unmapped_count} rows with no ImageNet mapping.")


def main() -> None:
    _data_dir = Path(
        os.getenv("EVAL_DATA_DIR", Path(__file__).resolve().parent.parent / "data")
    ) / "pug_imagenet"

    parser = argparse.ArgumentParser(
        description="Convert PUG-ImageNet labels CSV to Metadata_Expanded.csv."
    )
    parser.add_argument(
        "--data-dir", default=str(_data_dir),
        help="Directory containing labels_pug_imagenet.csv and class_to_imagenet_idx.json "
             "(default: $EVAL_DATA_DIR/pug_imagenet or <repo>/data/pug_imagenet).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path. Defaults to <data-dir>/Metadata_Expanded.csv.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    source_csv = data_dir / "labels_pug_imagenet.csv"
    mapping_json = data_dir / "class_to_imagenet_idx.json"
    output_csv = Path(args.output) if args.output else data_dir / "Metadata_Expanded.csv"

    for path, name in [
        (source_csv, "labels CSV"),
        (mapping_json, "class mapping JSON"),
    ]:
        if not path.exists():
            print(f"Error: {name} not found at: {path}")
            return

    preprocess_pug(source_csv, mapping_json, output_csv)


if __name__ == "__main__":
    main()
