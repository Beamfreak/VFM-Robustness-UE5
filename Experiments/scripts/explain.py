"""
Explainability script for Vision Model Evaluation Framework.

Uses Class Activation Mapping (Grad-CAM) to visualize where the model looks
when making a classification decision. Supports both CNNs (e.g., ResNet) and 
Vision Transformers (e.g., ViT, Swin) loaded via timm.

Requirements:
    pip install grad-cam opencv-python
"""

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM, EigenCAM, ScoreCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    print("Please install required packages first:")
    print("pip install grad-cam opencv-python")
    exit(1)

from .config import Config, MODELS
from .models import ModelLoader
from .utils import load_image, set_seed, load_imagenet_index

def get_target_layer(model: torch.nn.Module, model_key: str):
    """
    Returns the appropriate target layer for Grad-CAM depending on the model architecture.
    """
    model_key = model_key.lower()
    
    # ResNet-style architectures
    if "resnet" in model_key:
        if hasattr(model, "layer4"):
            return [model.layer4[-1]]
        if hasattr(model, "encoder") and hasattr(model.encoder, "stages"):
            try:
                return [model.encoder.stages[-1].layers[-1].layer[-1]]
            except Exception:
                pass
        conv_layers = [module for _, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
        if conv_layers:
            return [conv_layers[-1]]
    
    # ConvNeXt architectures
    elif "convnext" in model_key:
        return [model.stages[-1][-1]]
    
    # Vision Transformers (ViT, DINO, CLIP)
    elif "vit" in model_key or "dino" in model_key or "clip" in model_key:
        if hasattr(model, "blocks"):
            # timm style ViT (e.g. DINOv3 Eva, CLIP-ViT)
            return [model.blocks[-1].norm1]
        elif hasattr(model, "dinov2"):
            # HuggingFace DINOv2
            return [model.dinov2.encoder.layer[-1].norm1]
        elif hasattr(model, "vit") and hasattr(model.vit, "encoder"):
            # HuggingFace style ViT
            return [model.vit.encoder.layer[-1].layernorm_before]
        else:
            return [list(model.children())[-1]]
    
    # Swin Transformers
    elif "swin" in model_key:
        return [model.layers[-1].blocks[-1].norm1]
    
    else:
        # Fallback (may or may not work depending on the exact architecture)
        print(f"Warning: Standard target layer unknown for {model_key}. Attempting to use the last layer.")
        return [list(model.children())[-1]]


class HuggingFaceModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "logits"):
            return out.logits
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        if isinstance(out, tuple):
            return out[0]
        return out

def setup_cam_model(bundle, model_key):
    """Initializes and returns the CAM algorithm and wrapped model for a given run."""
    cam_model = HuggingFaceModelWrapper(bundle.model)

    # 3. Get architectural target layer
    target_layers = get_target_layer(bundle.model, model_key)
    if not target_layers:
        return None, None
    
    # For ViT/Transformers, GradCAM requires reshaping the flattened patch tokens.
    reshape_transform = None
    if "swin" in model_key:
        def swin_reshape_transform(tensor):
            return tensor.permute(0, 3, 1, 2)
        reshape_transform = swin_reshape_transform
    elif "vit" in model_key or "dino" in model_key or "clip" in model_key:
        def vit_reshape_transform(tensor, height=14, width=14):
            expected_tokens = height * width
            extra_tokens = tensor.shape[1] - expected_tokens
            if extra_tokens > 0:
                tensor = tensor[:, extra_tokens:, :]
            result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        
        def dynamic_vit_reshape(tensor):
            seq_len = tensor.shape[1]
            possible_patch_counts = [x*x for x in range(1, 100)]
            for patches in reversed(possible_patch_counts):
                if seq_len >= patches and (seq_len - patches) < 10:
                    grid_dim = int(np.sqrt(patches))
                    return vit_reshape_transform(tensor, grid_dim, grid_dim)
            return vit_reshape_transform(tensor, 14, 14)
            
        reshape_transform = dynamic_vit_reshape

    CAM_ALGO = GradCAM
    try:
        cam = CAM_ALGO(
            model=cam_model, 
            target_layers=target_layers, 
            reshape_transform=reshape_transform
        )
        return cam, cam_model
    except Exception as e:
        print(f"[{model_key}] Error initializing CAM: {e}")
        return None, None

def process_single_image(args, config, device, model_loader):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]  # BGR to RGB
    rgb_img = np.float32(rgb_img) / 255
    pil_img = Image.fromarray(np.uint8(rgb_img * 255))
    
    try:
        imagenet_labels = load_imagenet_index(config.paths.imagenet_index_path)
    except Exception:
        imagenet_labels = {}

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"Skipping unknown model: {model_key}")
            continue
            
        print(f"\n[{model_key}] Initializing explainability...")
        bundle = model_loader.load_model(model_key=model_key, override_eval_mode="logits")
        bundle.model.eval()

        cam, cam_model = setup_cam_model(bundle, model_key)
        if not cam:
            continue

        try:
            input_tensor = bundle.transform(pil_img).unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]

            with torch.no_grad():
                out = cam_model(input_tensor)
                pred_idx = out.argmax(dim=1).item()

            img_size = bundle.model_spec.input_size
            vis_img = cv2.resize(rgb_img, (img_size, img_size))

            visualization = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)

            label_name = imagenet_labels.get(pred_idx, str(pred_idx))
            clean_label = str(label_name).replace(" ", "_").replace(",", "").replace("/", "-")
            
            out_file = out_dir / f"{model_key}_pred{pred_idx}_{clean_label}_cam.jpg"
            cv2.imwrite(str(out_file), visualization[:, :, ::-1])
            print(f"[{model_key}] Heatmap saved to: {out_file} (Predicted: {label_name})")

        except Exception as e:
            print(f"[{model_key}] Error generating CAM: {e}")

def process_dataset_averages(args, config, device, model_loader):
    from tqdm import tqdm
    
    out_dir = Path(args.output_dir) / "averages"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = config.paths.metadata_path
    if not metadata_path.exists():
        print(f"Metadata file not found at {metadata_path}! Cannot run dataset analysis.")
        return
        
    from .data_loader import load_preprocessed_metadata
    df = load_preprocessed_metadata(metadata_path)
    image_root = config.paths.image_root
    
    for model_key in args.models:
        if model_key not in MODELS: continue
        
        print(f"\n[{model_key}] Generating average heatmaps...")
        bundle = model_loader.load_model(model_key=model_key, override_eval_mode="logits")
        bundle.model.eval()
        
        cam, _ = setup_cam_model(bundle, model_key)
        if not cam: continue
        
        for folder_name in args.group_vars:
            # Map user's requested folder name back to the actual CSV column name
            folder_to_col_mapping = {
                "Background": "Level",
                "Material": "Material",
                "Camera": "Camera Position",
                "Light": "Light Color (RGB)",
                "Fog": "Fog"
            }
            
            var = folder_to_col_mapping.get(folder_name)
            
            if not var or var not in df.columns:
                print(f"Skipping '{folder_name}': not mapped to a valid metadata column.")
                continue
            
            # Extract just the relevant subset of the dataframe that tests this variable
            # by matching the exact folder name at the beginning of the image path
            var_df = df[df['Image'].str.contains(rf"^{folder_name}/|/{folder_name}/", case=False, na=False)]
            
            if len(var_df) == 0:
                print(f"  -> No images found containing '{folder_name}' in their path. Falling back to full dataset.")
                var_df = df

            groups = var_df.groupby(var)
            for val, group_df in groups:
                # If filter_vals is provided, skip values not in the list
                if args.filter_vals and str(val) not in args.filter_vals:
                    continue

                if args.max_samples_per_group > 0:
                    sample_df = group_df.sample(min(len(group_df), args.max_samples_per_group), random_state=config.eval.seed)
                else:
                    sample_df = group_df
                
                accumulated_cam = None
                count = 0
                
                print(f"  -> Processing {var} = {val} ({len(sample_df)} samples)")
                
                # We'll compute the average size for resizing
                img_size = bundle.model_spec.input_size

                for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Images", leave=False):
                    img_path = image_root / str(row['Image'])
                    if not img_path.exists():
                        print(f"Image not found: {img_path}")
                        continue
                    
                    try:
                        # Load using PIL to apply transformations (which resize/crop accurately)
                        pil_img = Image.open(img_path).convert("RGB")
                        input_tensor = bundle.transform(pil_img).unsqueeze(0).to(device)
                        
                        # Generate CAM
                        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                        cam_map = grayscale_cam[0, :]
                        
                        # Resize the raw cam activation to match the input size standard
                        # (in case the model is fully conv and allows dynamic sized images)
                        cam_resized = cv2.resize(cam_map, (img_size, img_size))
                        
                        if accumulated_cam is None:
                            accumulated_cam = np.zeros_like(cam_resized, dtype=np.float32)
                            
                        accumulated_cam += cam_resized
                        count += 1
                        
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Error processing image {img_path}: {e}")
                        pass
                        
                if count > 0:
                    avg_cam = accumulated_cam / count
                    
                    # Normalize correctly into 0-1 for heatmap rendering
                    avg_cam_norm = (avg_cam - avg_cam.min()) / (avg_cam.max() - avg_cam.min() + 1e-8)
                    
                    # Apply ColorMap
                    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * avg_cam_norm), cv2.COLORMAP_JET)
                    
                    clean_val = str(val).replace(" ", "_").replace("/", "-").replace("\\", "-")
                    clean_var = str(var).replace(" ", "_")
                    file_path = out_dir / f"{model_key}_{clean_var}_{clean_val}_avg.jpg"
                    
                    cv2.imwrite(str(file_path), heatmap_bgr)
            
            print(f"Saved average CAMs for {model_key} grouped by {var} in {out_dir}")


#def process_object_analysis(args, config, device, model_loader):

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explainability heatmaps")
    parser.add_argument("--models", nargs="+", help="Models to evaluate (e.g. resnet50 vit_b). Defaults to all models if not provided.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a specific image to visualize")
    parser.add_argument("--dataset_analysis", action="store_true", help="Calculate average CAMs per dataset metadata variable")
    parser.add_argument("--analyze_object", type=str, default=None, help="Generate an overall average heatmap and individual heatmaps per variation for a specific object class (e.g., wine-bottle-1)")
    parser.add_argument("--fidelity", action="store_true", help="Calculate fidelity scores (Confidence Drop and Retention) for the generated heatmaps.")
    parser.add_argument("--group_vars", nargs="+", default=["Background", "Material", "Camera", "Light", "Fog"], help="Folder groups to process for average CAMs")
    parser.add_argument("--filter_vals", nargs="+", default=None, help="Optional: Only process these specific variable values (e.g. 'M_Glass' '30_90')")
    parser.add_argument("--max_samples_per_group", type=int, default=-1, help="Max images to average per group (-1 for all)")
    parser.add_argument("--output_dir", type=str, default="results/explainability", help="Output directory")
    args = parser.parse_args()

    if not args.image_path and not args.dataset_analysis and not args.analyze_object:
        parser.error("Must specify either --image_path, --dataset_analysis, or --analyze_object.")

    if not args.models:
        args.models = list(MODELS.keys())
        print(f"No models specified. Defaulting to all models: {', '.join(args.models)}")

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.eval.seed)

    model_loader = ModelLoader(device=device)

    if args.image_path:
        process_single_image(args, config, device, model_loader)
        
    if args.dataset_analysis:
        process_dataset_averages(args, config, device, model_loader)
        
    if args.analyze_object:
        process_object_analysis(args, config, device, model_loader)

if __name__ == "__main__":
    main()