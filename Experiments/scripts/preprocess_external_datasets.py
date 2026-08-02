import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import tempfile
import re
import ast
import difflib

import json

# Locate the external data_scripts package. Set EVAL_DATA_SCRIPTS_DIR to point at the
# directory that contains the 'experiments' package, or let it fall back to a sibling
# directory named 'data_scripts' next to this script's parent folder.
_SCRIPTS_ROOT = os.getenv(
    "EVAL_DATA_SCRIPTS_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data_scripts")
)
tmpdir = tempfile.mkdtemp()
os.symlink(_SCRIPTS_ROOT, os.path.join(tmpdir, "experiments"))
sys.path.insert(0, tmpdir)

try:
    from experiments.data._factory import create_dataset
    from experiments.data._constants import DATASET_SPLITS
except ImportError as e:
    print(f"Error loading external scripts: {e}")
    sys.exit(1)


def load_imagenet_mapping(json_path):
    with open(json_path) as f:
        mapping = json.load(f)
    synset_to_idx = {v[0]: int(k) for k, v in mapping.items()}
    synset_to_label = {v[0]: v[1] for k, v in mapping.items()}
    idx_to_label = {int(k): v[1] for k, v in mapping.items()}
    return synset_to_idx, synset_to_label, idx_to_label


def _normalize_label(text):
    if not isinstance(text, str):
        return ""
    text = text.strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_eval_framework_imagenet_index():
    """Load idx->label mapping from eval-framework imagenet_class_index.txt when available."""
    local_index = Path(__file__).resolve().parent.parent / "imagenet_class_index.txt"
    if not local_index.exists():
        return {}

    try:
        content = local_index.read_text(encoding="utf-8")
        parsed = ast.literal_eval(content)
        return {int(k): str(v) for k, v in parsed.items()}
    except Exception:
        return {}


def load_imagenet_d_gt_labels(data_dir):
    """Load ImageNet-D category -> GT label from questions files."""
    questions_dir = Path(data_dir) / "imagenet-d" / "questions"
    if not questions_dir.exists():
        return {}

    category_to_gt = {}
    for csv_path in questions_dir.glob("*.csv"):
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("image_dir"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                image_rel = parts[0]
                gt_label = parts[1].strip()
                image_parts = image_rel.split("/")
                if len(image_parts) >= 2 and gt_label:
                    category_to_gt[image_parts[1]] = gt_label
    return category_to_gt


def _split_label_aliases(label_text):
    return [_normalize_label(x) for x in str(label_text).split(",") if _normalize_label(x)]


def _candidate_match_score(category_name, gt_label, candidate_label):
    """Score candidate ImageNet label for a category using exact/alias/token/sequence cues."""
    queries = []
    if gt_label:
        gt_norm = _normalize_label(gt_label)
        gt_no_paren = _normalize_label(re.sub(r"\([^)]*\)", "", gt_label))
        if gt_norm:
            queries.append(gt_norm)
        if gt_no_paren and gt_no_paren not in queries:
            queries.append(gt_no_paren)

    cat_norm = _normalize_label(category_name.replace("_", " "))
    if cat_norm and cat_norm not in queries:
        queries.append(cat_norm)

    aliases = _split_label_aliases(candidate_label)
    if not queries or not aliases:
        return 0.0

    def token_f1(a, b):
        ta = set(a.split())
        tb = set(b.split())
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        return (2.0 * inter) / (len(ta) + len(tb))

    best = 0.0
    for q in queries:
        for alias in aliases:
            if q == alias:
                return 1.0
            seq = difflib.SequenceMatcher(None, q, alias).ratio()
            tok = token_f1(q, alias)
            score = 0.65 * tok + 0.35 * seq
            if score > best:
                best = score
    return best


def load_imagenet_d_category_mapping(data_dir, idx_to_label):
    """Build ImageNet-D category -> (global idx, label) mapping.

    Priority:
    1) Explicit category->candidate_indices from imgnet_d2imgnet_id.txt
    2) Resolve multi-candidate classes with GT labels from questions/*.csv
    3) Fallback to first candidate index
    """
    imagenet_d_root = Path(data_dir) / "imagenet-d"
    explicit_map_path = imagenet_d_root / "imgnet_d2imgnet_id.txt"
    if not explicit_map_path.exists():
        return {}

    eval_idx_to_label = load_eval_framework_imagenet_index()
    label_space = eval_idx_to_label if eval_idx_to_label else idx_to_label
    category_to_gt = load_imagenet_d_gt_labels(data_dir)

    try:
        raw_text = explicit_map_path.read_text(encoding="utf-8").strip()
        try:
            category_to_candidates = json.loads(raw_text)
        except json.JSONDecodeError:
            category_to_candidates = ast.literal_eval(raw_text)
    except Exception:
        return {}

    category_to_idx_label = {}
    unresolved = 0
    multi_choice = 0

    for category, candidates in category_to_candidates.items():
        if not isinstance(candidates, list) or len(candidates) == 0:
            unresolved += 1
            continue

        candidate_ids = [int(c) for c in candidates]
        gt_label = category_to_gt.get(category, "")

        best_idx = None
        best_score = -1.0
        for idx in candidate_ids:
            cand_label = label_space.get(idx)
            if not cand_label:
                continue
            score = _candidate_match_score(category, gt_label, cand_label)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            best_idx = candidate_ids[0]

        if len(candidate_ids) > 1:
            multi_choice += 1

        final_label = label_space.get(best_idx, idx_to_label.get(best_idx, f"class_{best_idx}"))
        category_to_idx_label[category] = (int(best_idx), final_label)

    print(f"Loaded explicit ImageNet-D mapping for {len(category_to_idx_label)} categories from {explicit_map_path}.")
    if multi_choice > 0:
        print(f"  Resolved {multi_choice} multi-candidate categories using label matching.")
    if unresolved > 0:
        print(f"  Warning: {unresolved} categories had empty/invalid candidate lists.")

    return category_to_idx_label


def get_image_path(dataset_instance, idx):
    """Attempt to extract the actual image path from an underlying PyTorch dataset."""
    
    def extract_from_ds(ds, current_idx):
        # Base checks for direct path attributes
        if hasattr(ds, "samples"): return ds.samples[current_idx][0]
        elif hasattr(ds, "imgs"): return ds.imgs[current_idx][0]
        elif hasattr(ds, "image_paths"): return ds.image_paths[current_idx]
        elif hasattr(ds, "_filepaths"): return ds._filepaths[current_idx]
        
        # Handle HF Datasets
        if hasattr(ds, "hf_dataset") and hasattr(ds.hf_dataset, "data"):
            try:
                # Direct PyArrow access avoids loading/decoding the image bytes
                img_struct = ds.hf_dataset.data.column("image")[current_idx].as_py()
                if img_struct and "path" in img_struct and img_struct["path"]:
                    return img_struct["path"]
            except Exception:
                pass
                
        # Handle ConcatDataset
        if hasattr(ds, "datasets") and hasattr(ds, "cumulative_sizes"):
            import bisect
            dataset_idx = bisect.bisect_right(ds.cumulative_sizes, current_idx)
            sample_idx = current_idx if dataset_idx == 0 else current_idx - ds.cumulative_sizes[dataset_idx - 1]
            return extract_from_ds(ds.datasets[dataset_idx], sample_idx)
            
        # Unwrap standard wrappers/decorators
        if hasattr(ds, "dataset"):
            return extract_from_ds(ds.dataset, current_idx)
            
        return None

    return extract_from_ds(dataset_instance, idx)

def get_target(dataset_instance, idx):
    """Attempt to extract the target label from an underlying PyTorch dataset."""
    
    def extract_from_ds(ds, current_idx):
        if hasattr(ds, "targets"): return ds.targets[current_idx]
        if hasattr(ds, "samples"): return ds.samples[current_idx][1]
        if hasattr(ds, "imgs"): return ds.imgs[current_idx][1]
        
        # Handle HF Datasets
        if hasattr(ds, "hf_dataset") and hasattr(ds.hf_dataset, "data"):
            try:
                return ds.hf_dataset.data.column("label")[current_idx].as_py()
            except Exception:
                pass

        # Handle ConcatDataset
        if hasattr(ds, "datasets") and hasattr(ds, "cumulative_sizes"):
            import bisect
            dataset_idx = bisect.bisect_right(ds.cumulative_sizes, current_idx)
            sample_idx = current_idx if dataset_idx == 0 else current_idx - ds.cumulative_sizes[dataset_idx - 1]
            return extract_from_ds(ds.datasets[dataset_idx], sample_idx)
            
        # Unwrap standard wrappers/decorators
        if hasattr(ds, "dataset"):
            return extract_from_ds(ds.dataset, current_idx)
            
        return None

    return extract_from_ds(dataset_instance, idx)


def preprocess_dataset(dataset_name, data_dir, output_dir, synset_to_idx, synset_to_label, idx_to_label):
    """
    Load a dataset using the imported factory, extract all paths and classes, 
    and save them into the standard Metadata_Expanded.csv format.
    """
    if dataset_name == "imagenet_mini":
        splits = ["train", "val", "test"]
    else:
        splits = DATASET_SPLITS.get(dataset_name)
        
    if not splits:
        print(f"Skipping {dataset_name}: Unknown dataset splits.")
        return
        
    print(f"====================================")
    print(f"Preprocessing {dataset_name}...")
    print(f"====================================")
    
    dataset_records = []
    imagenet_d_category_map = {}
    if dataset_name == "imagenet_d":
        imagenet_d_category_map = load_imagenet_d_category_mapping(data_dir, idx_to_label)
        if imagenet_d_category_map:
            print(f"Loaded ImageNet-D category mapping for {len(imagenet_d_category_map)} categories.")
    
    # We mainly map 'test' or 'val'. For image_9 and image_d, they use different keys
    split_to_use = "test"
    if "test" not in splits and "val" in splits:
        split_to_use = "val"
    elif "test" not in splits and "all" in splits:
        split_to_use = "all"
        
    try:
        kwargs = {}
        if dataset_name == "objectnet":
            kwargs["objectnet_overlapping_only"] = True
            kwargs["objectnet_use_imagenet_labels"] = True
            
        # Load the dataset
        factory_ds_name = "mini_imagenet" if dataset_name == "imagenet_mini" else dataset_name
        ds = create_dataset(factory_ds_name, data_dir, split=split_to_use, **kwargs)
        
        unmapped_count = 0
        total_items = len(ds)
        print(f"Loaded {total_items} items.")
        
        images_dump_dir = os.path.join(output_dir, dataset_name.replace("_", "-"), "extracted_images")
        os.makedirs(images_dump_dir, exist_ok=True)

        for idx in range(total_items):
            try:
                # Try to get path and target without loading the image tensor
                path = get_image_path(ds, idx)
                target = get_target(ds, idx)
                
                dumped_filename = f"{idx:08d}.png"
                dumped_path = os.path.join(images_dump_dir, dumped_filename)
                
                is_valid_dump = os.path.exists(dumped_path) and os.path.getsize(dumped_path) > 100

                # If target is None, we likely failed to extract it efficiently and need to load the full item
                if target is None:
                    img_obj, target, original_idx = ds[idx]
                elif (not path or not os.path.exists(path)) and not is_valid_dump:
                    # Target was found but no real file exists (like in some HF datasets), we need to extract and save the image
                    img_obj, target, original_idx = ds[idx]
            except Exception as read_err:
                print(f"Failed reading item {idx}: {read_err}")
                continue

            if not path or not os.path.exists(path):
                if not is_valid_dump:
                    # Handle datasets that don't expose paths (like HF arrow cache)
                    if hasattr(img_obj, "save"):
                        img_obj.save(dumped_path)
                    else:
                        import torchvision
                        torchvision.utils.save_image(img_obj, dumped_path)
                path = dumped_path
                
            # Make the path relative to the image_root which we will set as the dataset's base directory
            dataset_root = str(Path(data_dir) / dataset_name.replace("_", "-"))
            output_dataset_root = str(Path(output_dir) / dataset_name.replace("_", "-"))

            if path.startswith(output_dataset_root):
                rel_path = os.path.relpath(path, output_dataset_root)
            elif path.startswith(dataset_root):
                rel_path = os.path.relpath(path, dataset_root)
            elif path.startswith(output_dir):
                rel_path = os.path.relpath(path, output_dir)
            elif path.startswith(data_dir):
                rel_path = os.path.relpath(path, data_dir)
            else:
                rel_path = path  # Fallback to absolute
                
            folder_name = os.path.basename(os.path.dirname(path))
            true_class = -1
            label_name = "<unmapped>"
            
            if dataset_name == "imagenet_d" and folder_name in imagenet_d_category_map:
                true_class, label_name = imagenet_d_category_map[folder_name]
            elif dataset_name == "imagenet_9":
                # imagenet_9 files often contain the true synset ID like n02085620_16757.JPEG
                synset_match = re.search(r'(n\d{8})', os.path.basename(path))
                if synset_match and synset_match.group(1) in synset_to_idx:
                    true_class = synset_to_idx[synset_match.group(1)]
                    label_name = synset_to_label[synset_match.group(1)]
                else:
                    true_class = int(target)
                    label_name = idx_to_label.get(true_class, f"class_{true_class}")
            elif folder_name.isdigit():
                true_class = int(folder_name)
                label_name = idx_to_label.get(true_class, f"class_{true_class}")
            elif folder_name in synset_to_idx:
                true_class = synset_to_idx[folder_name]
                label_name = synset_to_label[folder_name]
            else:
                # If we cannot map by folder, use the target generated by PyTorch, 
                # although it might be wrong due to alphabetical sorting.
                
                # Check HF datasets internal int2str mappings
                # Unwrap dataset if wrapped
                current_ds = ds
                while hasattr(current_ds, "dataset"):
                    current_ds = current_ds.dataset
                
                if hasattr(current_ds, "hf_dataset") and hasattr(current_ds.hf_dataset.features.get("label"), "int2str"):
                    # This maps local subset ID (0-99) into real synsets (n01532829)
                    hf_synset = current_ds.hf_dataset.features["label"].int2str(int(target))
                    # Handle if the dataset returns a direct string mapping not matching the index
                    import json
                    # Special check since we know imagenet-mini returns like "n01532829"
                    
                    if hf_synset in synset_to_idx:
                        true_class = synset_to_idx[hf_synset]
                        label_name = synset_to_label[hf_synset]
                    else:
                        # Some versions might return string indices
                        found_class = False
                        for idx_str, aliases in idx_to_label.items():
                             if type(aliases) is str and aliases.startswith(hf_synset):
                                 true_class = idx_str
                                 label_name = aliases
                                 found_class = True
                                 break
                        if not found_class:
                            true_class = int(target)
                            label_name = idx_to_label.get(true_class, f"class_{true_class}")
                else:
                    true_class = int(target)
                    label_name = idx_to_label.get(true_class, f"class_{true_class}")
                
            entry = {
                "Image": rel_path,
                "Object": "<unmapped>",
                "Level": "<unmapped>",
                "Class": true_class,
                "Material": "<unmapped>",
                "Camera Position": "<unmapped>",
                "Light Color (RGB)": "<unmapped>",
                "Fog": "<unmapped>",
                "ImageNet_Label": label_name,
                "ShapeNet_Superclass": "<unmapped>",
                "Source_Dataset": dataset_name,
                "Source_Split": split_to_use
            }
            dataset_records.append(entry)
            
        print(f"Processed paths! {len(dataset_records)} successfully located. {unmapped_count} paths could not be determined.")
        
        if len(dataset_records) > 0:
            df = pd.DataFrame(dataset_records)
            
            # Map Dataset name onto the path exactly how it appears usually
            dataset_folder = dataset_name.replace("_", "-") 
            out_folder = Path(output_dir) / dataset_folder
            out_folder.mkdir(parents=True, exist_ok=True)
            
            out_csv = out_folder / "Metadata_Expanded.csv"
            # Keep separator strictly to ';' as expected by our loader
            df.to_csv(out_csv, index=False, sep=";")
            print(f"✅ Successfully wrote Metadata to {out_csv}\n")
            
    except Exception as e:
        print(f"❌ Failed to preprocess {dataset_name}: {e}\n")


if __name__ == "__main__":
    _default_data_dir = os.getenv("EVAL_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

    parser = argparse.ArgumentParser(
        description="Preprocess external ImageNet-variant datasets into Metadata_Expanded.csv."
    )
    parser.add_argument("--data_dir", type=str, default=_default_data_dir,
                        help="Root directory where datasets are stored (default: $EVAL_DATA_DIR or <repo>/data).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for output CSVs. Defaults to --data_dir if not set.")
    parser.add_argument("--datasets", nargs="+", default=["imagenet_a", "imagenet_r", "imagenet_v2"],
                        help="Dataset names to preprocess.")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir

    # The class-index JSON is expected under <data_dir>/imagenet-a/ by convention.
    mapping_file = str(Path(args.data_dir) / "imagenet-a" / "imagenet_class_index.json")
    synset_to_idx, synset_to_label, idx_to_label = load_imagenet_mapping(mapping_file)

    for ds_name in args.datasets:
        ds_name = ds_name.replace("-", "_")
        if ds_name == "imagenet_tiny":
            ds_name = "tiny_imagenet"
        preprocess_dataset(ds_name, args.data_dir, args.output_dir, synset_to_idx, synset_to_label, idx_to_label)