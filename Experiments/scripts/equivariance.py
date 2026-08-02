import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from tqdm import tqdm

from scripts.config import DATASETS, get_default_config, MODELS
from scripts.models import ModelLoader
from scripts.knn import FeatureExtractor
from scripts.data_loader import create_dataloader, load_preprocessed_metadata

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def measure_equivariance(embeddings: dict, metadata: pd.DataFrame, target_factor: str) -> float:
    """
    Measure the representation equivariance of embeddings with respect to a target factor.

    Args:
        embeddings: A dictionary mapping image IDs to their embedding vectors.
        metadata: A pandas DataFrame containing metadata for each image.
        target_factor: The column name in metadata to evaluate.

    Returns:
        The mean cosine similarity averaged over all transitions and all object pairs.
    """
    # Filter metadata to only include images with embeddings
    available_ids = set(embeddings.keys())
    metadata = metadata[metadata['image_id'].isin(available_ids)]

    # Identify all columns other than target_factor and image_id to use for grouping
    # Assuming 'object_id' or 'class_id' exists. Let's use 'object_id' if available, otherwise 'class_id'.
    object_col = 'object_id'
    if object_col not in metadata.columns:
        if 'class_id' in metadata.columns:
            object_col = 'class_id'
        else:
            raise ValueError("Could not find 'object_id' or 'class_id' in metadata.")

    # Group columns: everything except image_id, target_factor, and object_col
    group_cols = [col for col in metadata.columns if col not in ['image_id', target_factor, object_col]]

    # Find pairs of images for the same object where all group_cols are identical but target_factor differs
    similarity_scores = []
    
    # Iterate through each identical context group
    for context, group in metadata.groupby(group_cols, dropna=False):
        # Group by object within this context
        object_transitions = defaultdict(dict)
        for _, row in group.iterrows():
            obj_id = row[object_col]
            val = row[target_factor]
            img_id = row['image_id']
            if img_id in embeddings:
                object_transitions[obj_id][val] = embeddings[img_id]
                
        # Find all possible valid transitions explicitly avoiding NaNs
        unique_vals = [v for v in group[target_factor].unique() if pd.notna(v)]
        all_vals = sorted(list(set(unique_vals)), key=str)
        
        for val_a, val_b in combinations(all_vals, 2):
            # Find objects that have both val_a and val_b
            valid_objects = [obj for obj, vals in object_transitions.items() if val_a in vals and val_b in vals]
            
            if len(valid_objects) < 2:
                continue
                
            # Calculate difference vectors for each valid object
            diff_vectors = {}
            for obj in valid_objects:
                vec_a = np.array(object_transitions[obj][val_a])
                vec_b = np.array(object_transitions[obj][val_b])
                diff_vectors[obj] = vec_b - vec_a
                
            # Calculate cosine similarity for all pairs of objects
            for obj_i, obj_j in combinations(valid_objects, 2):
                vec_i = diff_vectors[obj_i]
                vec_j = diff_vectors[obj_j]
                
                # SciPy's cosine computes distance, so similarity is 1 - distance
                sim = 1.0
                if np.linalg.norm(vec_i) > 0 and np.linalg.norm(vec_j) > 0:
                    dist = cosine(vec_i, vec_j)
                    if not np.isnan(dist):
                         sim = 1.0 - dist
                similarity_scores.append(sim)
                
    if not similarity_scores:
        return 0.0
        
    return float(np.mean(similarity_scores))

def main():
    parser = argparse.ArgumentParser(description="Measure equivariance of models.")
    parser.add_argument("--dataset", type=str, default="normal", help="Dataset to evaluate (e.g. 'normal' or 'material').")
    parser.add_argument("--model", type=str, default="all", help="Model to evaluate (e.g. 'clip_b', or 'all').")
    parser.add_argument("--variable", type=str, default="all", help="Target factor to evaluate (e.g. 'Camera', or 'all').")
    parser.add_argument("--object_id", type=str, default="all", help="Specific object ID to evaluate, or 'all'.")
    args = parser.parse_args()
    
    dataset_aliases = {
        "normal": "normal",
        "normal_dataset": "normal",
        "plastic": "material",
        "plastic_dataset": "material",
        "material": "material",
    }
    dataset_key = dataset_aliases.get(args.dataset, args.dataset)
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {sorted(DATASETS.keys())}")

    dataset_spec = DATASETS[dataset_key]

    logger.info(
        f"Equivariance evaluation | Dataset: {dataset_spec.name} | Model: {args.model} | Variable: {args.variable} | Object: {args.object_id}"
    )

    cfg = get_default_config()

    # Determine models
    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    # Map aliases and determine variables
    context_vars = ["Level", "Camera Position", "Light Color (RGB)", "Fog", "Material"]
    alias_map = {"Camera": "Camera Position", "Light": "Light Color (RGB)"}
    
    if args.variable == "all":
        vars_to_run = context_vars
    else:
        vars_to_run = [alias_map.get(args.variable, args.variable)]
    
    # Setup dataset matching
    metadata = load_preprocessed_metadata(dataset_spec.metadata_path)
    if args.object_id != "all":
        metadata = metadata[metadata["Object"] == args.object_id]

    out_folder = cfg.paths.results_dir / "equivariance" / dataset_spec.name 
    out_json = out_folder / "equivariance_results.json"
    out_folder.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    table_data = defaultdict(dict)

    model_loader = ModelLoader(device=cfg.eval.device)

    for i, model_key in enumerate(models_to_run):
        if i > 0:
            logger.info("Delaying model load by 5 seconds to reduce rate-limits...")
            time.sleep(5)
            
        logger.info(f"--- Loading model {model_key} ---")
        if model_key not in MODELS:
            logger.error(f"Model {model_key} not recognized in configuration. Skipping.")
            continue
            
        spec = MODELS[model_key]
        spec.eval_mode = "knn"  # Force embeddings via feature extractor

        bundle = model_loader.load_model(model_key)
        dataloader = create_dataloader(
            metadata, 
            dataset_spec.image_root, 
            bundle.transform, 
            spec.batch_size, 
            num_workers=cfg.eval.num_workers
        )
        
        logger.info(f"Extracting feature vectors for {model_key}...")
        extractor = FeatureExtractor(bundle, use_amp=cfg.eval.use_amp)
        features, feature_meta = extractor.extract_features(dataloader)
        embeddings = dict(zip(feature_meta["image_path"], features))
        
        # Clean metadata namespace
        eval_metadata = metadata.copy()
        eval_metadata.rename(columns={"Image": "image_id", "Object": "object_id"}, inplace=True, errors="ignore")
        
        useful_cols = ["image_id", "object_id"] + context_vars
        eval_metadata = eval_metadata[[col for col in useful_cols if col in eval_metadata.columns]]
        
        for variable in vars_to_run:
            if variable not in eval_metadata.columns:
                logger.warning(f"Factor '{variable}' missing from metadata context, skipping.")
                continue
                
            logger.info(f"Evaluating representation equivariance towards {variable} ({model_key})")
            score = measure_equivariance(embeddings, eval_metadata, variable)
            
            logger.info(f">> Equivariance score ({model_key} | {variable}): {score:.4f}")
            res_key = f"{model_key}_{variable}"
            all_results[res_key] = score
            table_data[model_key][variable] = score
            
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=4)
        logger.info(f"Metrics saved safely to {out_json}")
        
    if args.object_id == "all" and table_data:
        print("\n=== Global Equivariance Summary ===")
        df_table = pd.DataFrame.from_dict(table_data, orient='index')
        print(df_table.to_string(float_format=lambda x: f"{x:.4f}"))
        
        out_csv = out_folder / "equivariance_results.csv"
        df_table.to_csv(out_csv)
        logger.info(f"Tabular grid saved to {out_csv}")

if __name__ == "__main__":
    main()
