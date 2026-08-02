# Metadata Preprocessing - README

## Overview

Before running any model evaluation, you **must** preprocess the metadata to:
1. Re-index and inject missing images from the directory structure on disk
2. Filter out unnecessary segmentation masks
3. Add `ImageNet_Label` - Human-readable ImageNet class name
4. Add `ShapeNet_Superclass` - Mapped ShapeNet superclass (or `<unmapped>`)

This preprocessing step is done **once**, and all subsequent evaluations use the expanded metadata.

---

## Why Preprocessing?

### Problem
The original `Metadata.csv` only contains:
- `Class` column with ImageNet index (e.g., 468)
- No human-readable labels
- No ShapeNet superclass mapping
- Missing records for dynamically rendered images on disk, and contains unused mask records.

During evaluation, every model would need to:
- Look up ImageNet labels repeatedly
- Map ImageNet indices to ShapeNet repeatedly
- Handle unmapped classes repeatedly

### Solution
**Preprocess once, evaluate many times:**
- Re-index missing images into the metadata
- Discard unused `_mask.png` generated images
- Add `ImageNet_Label` and `ShapeNet_Superclass` columns upfront
- Ground truth becomes immediately available
- Evaluation code simplified (no mapping logic needed)
- Faster evaluation (mapping done once, not per-model)

---

## What Gets Added

### Original Columns (8)
```
Image | Object | Level | Class | Material | Camera Position | Light Color (RGB) | Fog
```

### Added Columns (3)
```
ImageNet_Label | ShapeNet_Superclass | Mask
```

### Example Transformation

**Before:**
```csv
Image;Object;Level;Class;Material;Camera Position;Light Color (RGB);Fog
Dataset/468/Desert_A/Desert_A_Taxi_C_0_C0_L0_M0_F0.png;Taxi_C;Desert_A;468;Default;X=0.000 Y=1.000 Z=0.200;(R=1.000000,G=0.896086,B=0.322917,A=1.000000);false
```

**After:**
```csv
Image;Object;Level;Class;Material;Camera Position;Light Color (RGB);Fog;Mask;ImageNet_Label;ShapeNet_Superclass
Dataset/468/Desert_A/Desert_A_Taxi_C_0_C0_L0_M0_F0.png;Taxi_C;Desert_A;468;Default;X=0.000 Y=1.000 Z=0.200;(R=1.000000,G=0.896086,B=0.322917,A=1.000000);false;Dataset/468/Desert_A/Desert_A_Taxi_C_0_C0_L0_M0_F0_mask.png;cab, hack, taxi, taxicab;car
```

---

## How to Run

### Prerequisites
```bash
# Ensure you have required files:
# 1. Original metadata
data/<dataset_name>/Metadata.csv

# 2. ImageNet index (in project root)
imagenet_class_index.txt

# 3. ShapeNet mapping (in project root)
ShapeNet-ImageNet1k-Mapping-Indexed-subcategories4.json
```

### Run Preprocessing Script
```bash
python scripts/preprocess_metadata.py
```

### Expected Output
```
Loading data...
Checking for missing images across all variables on disk...
  Found 1202 total images physically on disk.
✓ Injecting 300 newly discovered images into metadata!
✓ Filtered out 150 mask images
✓ Loaded 1052 rows from metadata
✓ Loaded 1000 ImageNet classes
✓ Built mapping for 350 ImageNet classes to ShapeNet

Expanding metadata...

✓ Expansion complete!
  Total rows: 1052
  Mapped to ShapeNet: 1052 (100.0%)
  Unmapped: 0 (0.0%)
  ShapeNet superclasses present: 1
  Classes: ['car']

✓ Saved expanded metadata to: data/normal_dataset/Metadata_Expanded.csv

Validation checks:
  ✓ All ImageNet labels resolved

Sample rows (first 3):
                                               Image  Class               ImageNet_Label ShapeNet_Superclass
0  Dataset/468/Desert_A/Desert_A_Taxi_C_0_C0_L...    468  cab, hack, taxi, taxicab                 car
1  Dataset/468/Desert_A/Desert_A_Taxi_C_0_C1_L...    468  cab, hack, taxi, taxicab                 car
2  Dataset/468/Desert_A/Desert_A_Taxi_C_0_C2_L...    468  cab, hack, taxi, taxicab                 car

============================================================
Preprocessing complete! Ready for evaluation.
============================================================
```

---

## Output File

### Location
```
data/<dataset_name>/Metadata_Expanded.csv
```

### Schema (11 columns)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| Image | str | Relative image path | `Dataset/468/Desert_A/...` |
| Object | str | Object ID | `Taxi_C` |
| Level | str | Scene/level name | `Desert_A` |
| Class | int | ImageNet class index (0-999) | `468` |
| Material | str | Material name | `Default` |
| Camera Position | str | Camera coordinates | `X=0.000 Y=1.000 Z=0.200` |
| Light Color (RGB) | str | Light color tuple | `(R=1.0,G=0.89,B=0.32,A=1.0)` |
| Fog | bool | Fog enabled | `false` |
| **Mask** | str | The corresponding generated mask relative path | `Dataset/..._mask.png` |
| **ImageNet_Label** ⭐ | str | **Human-readable label** | `cab, hack, taxi, taxicab` |
| **ShapeNet_Superclass** ⭐ | str | **Mapped superclass** | `car` |

---

## Mapping Details

### ImageNet → ShapeNet Mapping
The mapping comes from `ShapeNet-ImageNet1k-Mapping-Indexed-subcategories4.json`:

**Example:**
```json
{
  "car": {
    "imagenet_class_indices": [407, 436, 468, 511, 609, 627, 656, 661, 717, 734, 751, 757, 817],
    "imagenet_label_candidates": [
      "ambulance", "beach wagon", "cab, hack, taxi", "convertible", ...
    ]
  }
}
```

### Unmapped Classes
ImageNet classes **not** in the mapping receive `<unmapped>`:
- Animal classes (0-150, 151-269, etc.)
- Other objects without ShapeNet equivalents

These are:
- Included in ImageNet metrics
- Excluded from ShapeNet metrics
- Tracked separately as diagnostic

---

## Troubleshooting

### Script Fails to Find Files
**Error:** `❌ Error: Metadata file not found`

**Solution:** Verify that your target dataset directory exists under `data/<dataset_name>/` and contains a `Metadata.csv` file.

### Unknown ImageNet Labels
**Warning:** `⚠ Warning: N rows have unknown ImageNet labels`

**Cause:** `Class` column contains invalid index (>999 or <0)

**Solution:** Inspect `Metadata.csv` for invalid Class values, fix source data

---

## Next Steps

After preprocessing completes:

1. Verify `Metadata_Expanded.csv` exists
2. Check validation output (all green ✓)
3. Review sample rows for correctness
4. **Proceed to evaluation** (see [README.md](../README.md))

---

**Important:** This preprocessing step is **mandatory** before running any model evaluation. All subsequent evaluation scripts assume `Metadata_Expanded.csv` exists.
