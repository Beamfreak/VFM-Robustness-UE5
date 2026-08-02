"""
Metadata Preprocessing Script

Expands the original Metadata.csv by adding:
1. ImageNet_Label: Human-readable ImageNet class name
2. ShapeNet_Superclass: Mapped ShapeNet superclass (or <unmapped>)

Usage:
    python preprocess_metadata.py
"""

import pandas as pd
import json
from pathlib import Path


def load_imagenet_index(index_path: str) -> dict:
    """
    Load ImageNet class index mapping.
    
    Args:
        index_path: Path to imagenet_class_index.txt
        
    Returns:
        Dict mapping index (int) to label (str)
    """
    with open(index_path, 'r') as f:
        content = f.read()
    
    # Parse the dictionary format
    imagenet_index = eval(content)
    
    return imagenet_index


def build_imagenet_to_shapenet_mapping(shapenet_mapping_path: str) -> dict:
    """
    Build reverse mapping: ImageNet index -> ShapeNet superclass.
    
    Args:
        shapenet_mapping_path: Path to ShapeNet-ImageNet mapping JSON
        
    Returns:
        Dict mapping ImageNet index (int) to ShapeNet class (str)
    """
    with open(shapenet_mapping_path, 'r') as f:
        shapenet_mapping = json.load(f)
    
    imagenet_to_shapenet = {}
    
    for shapenet_class, mapping_data in shapenet_mapping.items():
        imagenet_indices = mapping_data.get('imagenet_class_indices', [])
        for idx in imagenet_indices:
            imagenet_to_shapenet[idx] = shapenet_class
    
    return imagenet_to_shapenet


def inject_missing_metadata(metadata: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    """
    Find all rendered images on disk (across all variables) that are missing from Metadata.csv and inject them.
    Assumes standard dataset structure: Dataset_Renderer/Dataset/<Variable>/<Value>/<filename>.png
    """
    import os
    import re
    
    print("Checking for missing images across all variables on disk...")
    # Use the last three components of the path (Variable/Value/filename) to uniquely identify images
    metadata['temp_rel_path'] = metadata['Image'].apply(lambda x: '/'.join(str(x).replace('\\', '/').split('/')[-3:]))
    metadata['temp_actor_id'] = metadata['temp_rel_path'].str.extract(r'Actor_(\d+)_')
    
    # Build mapping from actor id to object name and default properties
    if not metadata['temp_actor_id'].isna().all():
        # Prefer BaseMap rows for default properties
        basemap_rows = metadata[metadata['Level'] == 'BaseMap']
        if not basemap_rows.empty:
            actor_map = basemap_rows.dropna(subset=['temp_actor_id']).groupby('temp_actor_id').first()[
                ['Object', 'Class', 'Level', 'Material', 'Camera Position', 'Light Color (RGB)', 'Fog']
            ].to_dict('index')
        else:
            actor_map = metadata.dropna(subset=['temp_actor_id']).groupby('temp_actor_id').first()[
                ['Object', 'Class', 'Level', 'Material', 'Camera Position', 'Light Color (RGB)', 'Fog']
            ].to_dict('index')
    else:
        actor_map = {}
    
    # Look for all valid images in subdirectories (Background, Camera, Fog, Light, Material, etc.)
    all_image_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    actual_images = []
    for d in all_image_dirs:
        # exclude meta dirs or files
        curr_imgs = list(d.glob("*/*.png"))
        actual_images.extend(curr_imgs)
        
    actual_images = [f for f in actual_images if not f.name.lower().endswith('_mask.png')]
    print(f"  Found {len(actual_images)} total images physically on disk.")
    
    existing_rel_paths = set(metadata['temp_rel_path'].tolist())
    new_rows = []
    
    for img_path in actual_images:
        val_dir = img_path.parent.name
        var_dir = img_path.parent.parent.name
        filename = img_path.name
        
        # Unique identifier
        rel_path = f"{var_dir}/{val_dir}/{filename}"
        
        if rel_path not in existing_rel_paths:
            match = re.search(r'Actor_(\d+)_(\d+)\.png$', filename)
            if not match:
                continue
                
            actor_id = match.group(1)
            class_id = int(match.group(2))
            
            if actor_id in actor_map:
                base_props = actor_map[actor_id].copy()
            else:
                base_props = {
                    'Object': f"unknown-actor-{actor_id}",
                    'Class': class_id,
                    'Level': 'BaseMap',
                    'Material': 'Default',
                    'Camera Position': 'X=-1.000 Y=0.000 Z=0.200',
                    'Light Color (RGB)': float('nan'),
                    'Fog': 'false'
                }
                
            # Adjust properties based on variable directory
            if var_dir == "Background":
                base_props['Level'] = val_dir
            else:
                base_props['Level'] = 'BaseMap'
                
            if var_dir == "Material":
                base_props['Material'] = val_dir
            elif var_dir == "Fog":
                base_props['Fog'] = 'true' if val_dir.lower() == 'withfog' else 'false'
            
            # Construct the identical format path for the 'Image' column
            img_url_path = f"../../../../../../Users/Stud/Documents/Unreal Projects/Dataset_Renderer/Dataset/{var_dir}/{val_dir}/{filename}"
            mask_url_path = img_url_path.replace('.png', '_mask.png')
            
            new_rows.append({
                'Image': img_url_path,
                'Object': base_props['Object'],
                'Level': base_props['Level'],
                'Class': class_id,
                'Material': base_props['Material'],
                'Camera Position': base_props['Camera Position'],
                'Light Color (RGB)': base_props['Light Color (RGB)'],
                'Fog': base_props['Fog'],
                'Mask': mask_url_path
            })
            
    metadata = metadata.drop(columns=['temp_rel_path', 'temp_actor_id'])
    
    if new_rows:
        print(f"✓ Injecting {len(new_rows)} newly discovered images into metadata!")
        new_df = pd.DataFrame(new_rows)
        metadata = pd.concat([metadata, new_df], ignore_index=True)
    else:
        print("  No missing images found.")
        
    return metadata


def expand_metadata(metadata_path: str, 
                    imagenet_index_path: str,
                    shapenet_mapping_path: str,
                    output_path: str) -> None:
    """
    Expand metadata with ImageNet labels and ShapeNet superclasses.
    
    Args:
        metadata_path: Path to original Metadata.csv
        imagenet_index_path: Path to imagenet_class_index.txt
        shapenet_mapping_path: Path to ShapeNet-ImageNet mapping JSON
        output_path: Path to save expanded metadata
    """
    print("Loading data...")
    
    # Load metadata (semicolon-delimited)
    metadata = pd.read_csv(metadata_path, sep=';')
    
    # --- Inject Missing Images ---
    dataset_dir = Path(metadata_path).parent
    metadata = inject_missing_metadata(metadata, dataset_dir)
    # -----------------------------
    
    # Filter out segmentation mask images
    mask_count = metadata['Image'].str.contains('_mask', case=False, na=False).sum()
    if mask_count > 0:
        print(f"✓ Filtered out {mask_count} mask images")
        metadata = metadata[~metadata['Image'].str.contains('_mask', case=False, na=False)].reset_index(drop=True)
        
    print(f"✓ Loaded {len(metadata)} rows from metadata")
    
    # Load ImageNet index
    imagenet_index = load_imagenet_index(imagenet_index_path)
    print(f"✓ Loaded {len(imagenet_index)} ImageNet classes")
    
    # Build ImageNet -> ShapeNet mapping
    imagenet_to_shapenet = build_imagenet_to_shapenet_mapping(shapenet_mapping_path)
    print(f"✓ Built mapping for {len(imagenet_to_shapenet)} ImageNet classes to ShapeNet")
    
    print("\nExpanding metadata...")
    
    # Add ImageNet_Label column
    metadata['ImageNet_Label'] = metadata['Class'].apply(
        lambda idx: imagenet_index.get(idx, f"<unknown:{idx}>")
    )
    
    # Add ShapeNet_Superclass column
    metadata['ShapeNet_Superclass'] = metadata['Class'].apply(
        lambda idx: imagenet_to_shapenet.get(idx, "<unmapped>")
    )
    
    # Report statistics
    total_rows = len(metadata)
    mapped_rows = (metadata['ShapeNet_Superclass'] != "<unmapped>").sum()
    unmapped_rows = total_rows - mapped_rows
    
    print(f"\n✓ Expansion complete!")
    print(f"  Total rows: {total_rows}")
    print(f"  Mapped to ShapeNet: {mapped_rows} ({mapped_rows/total_rows*100:.1f}%)")
    print(f"  Unmapped: {unmapped_rows} ({unmapped_rows/total_rows*100:.1f}%)")
    
    # Report ShapeNet superclasses present
    shapenet_classes = metadata[metadata['ShapeNet_Superclass'] != "<unmapped>"]['ShapeNet_Superclass'].unique()
    print(f"  ShapeNet superclasses present: {len(shapenet_classes)}")
    print(f"  Classes: {sorted(shapenet_classes)}")
    
    # Save expanded metadata
    metadata.to_csv(output_path, sep=';', index=False)
    print(f"\n✓ Saved expanded metadata to: {output_path}")
    
    # Validation checks
    print("\nValidation checks:")
    missing_labels = (metadata['ImageNet_Label'].str.startswith("<unknown:")).sum()
    if missing_labels > 0:
        print(f"  ⚠ Warning: {missing_labels} rows have unknown ImageNet labels")
    else:
        print(f"  ✓ All ImageNet labels resolved")
    
    # Show sample
    print("\nSample rows (first 3):")
    print(metadata[['Image', 'Class', 'ImageNet_Label', 'ShapeNet_Superclass']].head(3).to_string())


def main():
    """Parse arguments and run metadata expansion."""
    import argparse
    import os

    _base_dir = Path(__file__).resolve().parent.parent
    _default_data_dir = os.getenv("EVAL_DATA_DIR", str(_base_dir / "data"))

    parser = argparse.ArgumentParser(
        description="Expand Metadata.csv with ImageNet labels and ShapeNet superclasses."
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to input Metadata.csv (semicolon-delimited)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for the output Metadata_Expanded.csv. Defaults to <metadata_dir>/Metadata_Expanded.csv."
    )
    parser.add_argument(
        "--imagenet-index", default=str(_base_dir / "imagenet_class_index.txt"),
        help="Path to imagenet_class_index.txt."
    )
    parser.add_argument(
        "--shapenet-mapping", default=str(_base_dir / "ShapeNet-ImageNet1k-Mapping-Indexed-subcategories4.json"),
        help="Path to the ShapeNet-ImageNet mapping JSON."
    )
    args = parser.parse_args()

    metadata_path = args.metadata
    output_path = args.output or str(Path(metadata_path).parent / "Metadata_Expanded.csv")
    imagenet_index_path = args.imagenet_index
    shapenet_mapping_path = args.shapenet_mapping

    for path, name in [
        (metadata_path, "Metadata"),
        (imagenet_index_path, "ImageNet index"),
        (shapenet_mapping_path, "ShapeNet mapping"),
    ]:
        if not Path(path).exists():
            print(f"Error: {name} file not found at: {path}")
            return

    expand_metadata(
        metadata_path=metadata_path,
        imagenet_index_path=imagenet_index_path,
        shapenet_mapping_path=shapenet_mapping_path,
        output_path=output_path,
    )

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()

