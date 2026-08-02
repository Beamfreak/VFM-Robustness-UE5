"""
Configuration for Vision Model Evaluation Framework.

Defines all models, paths, and evaluation settings as specified in EXPERIMENT_SPECIFICATION.md.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Root of the repo (one level above this file)
_BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory: override with EVAL_DATA_DIR env var, otherwise defaults to <repo>/data/
DATA_DIR = Path(os.getenv("EVAL_DATA_DIR", _BASE_DIR / "data"))


# --- Model Definitions ---

@dataclass
class ModelSpec:
    """Specification for a single model."""
    name: str                 # Human-readable name
    model_id: str            # timm model identifier
    input_size: int          # Input resolution (square)
    batch_size: int          # Recommended batch size
    precision: str = "amp"   # Precision mode (amp = automatic mixed precision)
    eval_mode: str = "logits"  # "logits" for classification head, "knn" for feature extraction
    extra_eval_modes: Tuple[str, ...] = ()


# Models with pretrained ImageNet-1K classification heads
MODELS: Dict[str, ModelSpec] = {
    "clip_b_in1k": ModelSpec(
        name="CLIP-B-IN1K",
        model_id="vit_base_patch16_clip_224.laion2b_ft_in1k",
        input_size=224,
        batch_size=256,
        eval_mode="logits",
        extra_eval_modes=("linear_probe",),
    ),

    "clip_b": ModelSpec(
        name="CLIP-B",
        model_id="vit_base_patch16_clip_224.openai",
        input_size=224,
        batch_size=256,
        eval_mode="knn",
    ),
    
    "dinov2_b_in1k": ModelSpec(
        name="DINOv2-B-IN1K",
        model_id="facebook/dinov2-base-imagenet1k-1-layer",
        input_size=518,
        batch_size=128,
        eval_mode="logits",
        extra_eval_modes=("linear_probe",),
    ),

    "dinov2_b": ModelSpec(
        name="DINOv2-B",
        model_id="vit_base_patch14_dinov2.lvd142m",
        input_size=518,
        batch_size=64,
        eval_mode="knn",
    ),
    
    "dinov1_b": ModelSpec(
        name="DINOv1-B",
        model_id="vit_base_patch16_224.dino",
        input_size=224,
        batch_size=256,
        eval_mode="knn",
    ),
    
    "dinov3_b": ModelSpec(
        name="DINOv3-B",
        model_id="vit_base_patch16_dinov3.lvd1689m",
        input_size=256,
        batch_size=256,
        eval_mode="knn",
    ),
    
    "swin_b_in1k": ModelSpec(
        name="Swin-B-IN1K",
        model_id="swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        input_size=224,
        batch_size=256,
        eval_mode="logits",
        extra_eval_modes=("linear_probe",),
    ),

     "swin_b": ModelSpec(
        name="Swin-B",
        model_id="swin_base_patch4_window7_224.ms_in1k",
        input_size=224,
        batch_size=256,
        eval_mode="knn",
    ),
    
    "vit_b_in1k": ModelSpec(
        name="ViT-B-IN1K",
        model_id="google/vit-base-patch16-224",
        input_size=224,
        batch_size=256,
        eval_mode="logits",
    ),
    
    # === ResNet-50 (Baseline) ===
    "resnet50_in1k": ModelSpec(
        name="ResNet-50-IN1K",
        model_id="resnet50.a1_in1k",
        input_size=224,
        batch_size=256,
        eval_mode="logits",
    ),

    "resnet50": ModelSpec(
        name="ResNet-50",
        model_id="microsoft/resnet-50",
        input_size=224,
        batch_size=256,
        eval_mode="knn",
    ),

    "hiera_b_in1k": ModelSpec(
        name="Hiera_B",
        model_id="hiera_base_224.mae_in1k_ft_in1k",
        input_size=224,
        batch_size=256,
        eval_mode="logits",
    ),




    # === Smaller models ===

    "dinov2_s_in1k": ModelSpec(
        name="DINOv2-S-IN1K",
        model_id="facebook/dinov2-small-imagenet1k-1-layer",
        input_size=518,
        batch_size=128,
        eval_mode="logits",
    ),

    "dinov2_s": ModelSpec(
        name="DINOv2-S",
        model_id="vit_small_patch14_dinov2.lvd142m",
        input_size=518,
        batch_size=64,
        eval_mode="knn",
    ),

    "dinov1_s": ModelSpec(
        name="DINOv1-S",
        model_id="vit_small_patch16_224.dino",
        input_size=224,
        batch_size=256,
        eval_mode="knn",
    ),

    "dinov3_s": ModelSpec(
        name="DINOv3-S",
        model_id="vit_small_patch16_dinov3.lvd1689m",
        input_size=256,
        batch_size=256,
        eval_mode="knn",
    ),

    "swin_s_in1k": ModelSpec(
        name="Swin-S-IN1K",
        model_id="swin_small_patch4_window7_224.ms_in22k_ft_in1k",
        input_size=224,
        batch_size=256,
        eval_mode="logits",
    ),

    "swin_s": ModelSpec(
        name="Swin-S",
        model_id="swin_small_patch4_window7_224.ms_in1k",
        input_size=224,
        batch_size=256,
        eval_mode="knn",
    ),

    "hiera_s_in1k": ModelSpec(
        name="Hiera_S-IN1K",
        model_id="hiera_small_224.mae_in1k_ft_in1k",
        input_size=224,
        batch_size=256,
        eval_mode="logits",
    ),





    # === Larger models ===

    "clip_l_in1k": ModelSpec(
        name="CLIP-L-IN1K",
        model_id="vit_large_patch14_clip_224.laion2b_ft_in1k",
        input_size=224,
        batch_size=16,
        eval_mode="logits",
    ),

    "clip_l": ModelSpec(
        name="CLIP-L",
        model_id="vit_large_patch14_clip_224.openai",
        input_size=224,
        batch_size=16,
        eval_mode="knn",
    ),

    "dinov2_l_in1k": ModelSpec(
        name="DINOv2-L-IN1K",
        model_id="facebook/dinov2-large-imagenet1k-1-layer",
        input_size=518,
        batch_size=16,
        eval_mode="logits",
    ),

    "dinov2_l": ModelSpec(
        name="DINOv2-L",
        model_id="vit_large_patch14_dinov2.lvd142m",
        input_size=518,
        batch_size=16,
        eval_mode="knn",
    ),

    "dinov3_l": ModelSpec(
        name="DINOv3-L",
        model_id="vit_large_patch16_dinov3.lvd1689m",
        input_size=256,
        batch_size=16,
        eval_mode="knn",
    ),

    "swin_l_in1k": ModelSpec(
        name="Swin-L-IN1K",
        model_id="swin_large_patch4_window7_224.ms_in22k_ft_in1k",
        input_size=224,
        batch_size=16,
        eval_mode="logits",
    ),

    "swin_l": ModelSpec(
        name="Swin-L",
        model_id="swin_large_patch4_window7_224",
        input_size=224,
        batch_size=16,
        eval_mode="knn",
    ),
    
    "vit_l_in1k": ModelSpec(
        name="ViT-L-IN1K",
        model_id="google/vit-large-patch16-224",
        input_size=224,
        batch_size=16,
        eval_mode="logits",
    ),

    "hiera_l_in1k": ModelSpec(
        name="Hiera_L-IN1K",
        model_id="hiera_large_224.mae_in1k_ft_in1k",
        input_size=224,
        batch_size=16,
        eval_mode="logits",
    ),
    
    
}


# Baseline model for comparison
BASELINE_MODEL = "resnet50"


# --- Dataset Definitions ---

@dataclass
class DatasetSpec:
    """Specification for a dataset."""
    name: str                 # Human-readable name
    image_root: Path          # Root for image paths
    metadata_path: Path       # Path to metadata csv
    baseline_dir: str         # Substring to define the baseline images
    is_synthetic: bool        # Whether the dataset uses synthetic structure (ShapeNet etc)

DATASETS: Dict[str, DatasetSpec] = {
    "multi-factor": DatasetSpec(
        name="multi-factor",
        image_root=DATA_DIR / "multi-factor",
        metadata_path=DATA_DIR / "multi-factor" / "Metadata_Expanded.csv",
        baseline_dir="Camera/Angle0/",
        is_synthetic=True,
    ),
    "multi-color": DatasetSpec(
        name="multi-color",
        image_root=DATA_DIR / "multi-color",
        metadata_path=DATA_DIR / "multi-color" / "Metadata_Expanded.csv",
        baseline_dir="Camera/Default/",
        is_synthetic=True,
    ),
    "pug_imagenet": DatasetSpec(
        name="pug_imagenet",
        image_root=DATA_DIR / "pug_imagenet",
        metadata_path=DATA_DIR / "pug_imagenet" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=True,
    ),
    "imagenet_a": DatasetSpec(
        name="imagenet_a",
        image_root=DATA_DIR / "imagenet-a",
        metadata_path=DATA_DIR / "imagenet-a" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "imagenet_mini": DatasetSpec(
        name="imagenet_mini",
        image_root=DATA_DIR / "imagenet-mini",
        metadata_path=DATA_DIR / "imagenet-mini" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "imagenet_sketch": DatasetSpec(
        name="imagenet_sketch",
        image_root=DATA_DIR / "imagenet-sketch",
        metadata_path=DATA_DIR / "imagenet-sketch" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "imagenet_1k": DatasetSpec(
        name="imagenet_1k",
        image_root=DATA_DIR / "imagenet-1k",
        metadata_path=DATA_DIR / "imagenet-1k" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "imagenet_d": DatasetSpec(
        name="imagenet_d",
        image_root=DATA_DIR / "imagenet-d",
        metadata_path=DATA_DIR / "imagenet-d" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "imagenet_tiny": DatasetSpec(
        name="imagenet_tiny",
        image_root=DATA_DIR / "tiny-imagenet",
        metadata_path=DATA_DIR / "tiny-imagenet" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "imagenet_9": DatasetSpec(
        name="imagenet_9",
        image_root=DATA_DIR / "imagenet-9",
        metadata_path=DATA_DIR / "imagenet-9" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "imagenet_hard": DatasetSpec(
        name="imagenet_hard",
        image_root=DATA_DIR / "imagenet-hard",
        metadata_path=DATA_DIR / "imagenet-hard" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "imagenet_r": DatasetSpec(
        name="imagenet_r",
        image_root=DATA_DIR / "imagenet-r",
        metadata_path=DATA_DIR / "imagenet-r" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "imagenet_v2": DatasetSpec(
        name="imagenet_v2",
        image_root=DATA_DIR / "imagenet-v2",
        metadata_path=DATA_DIR / "imagenet-v2" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "objectnet": DatasetSpec(
        name="objectnet",
        image_root=DATA_DIR / "objectnet",
        metadata_path=DATA_DIR / "objectnet" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
    "imagenet_c": DatasetSpec(
        name="imagenet_c",
        image_root=DATA_DIR / "imagenet-c",
        metadata_path=DATA_DIR / "imagenet-c" / "Metadata_Expanded.csv",
        baseline_dir="",
        is_synthetic=False,
    ),
}

DEFAULT_DATASET = "normal"



# --- Path Configuration ---



@dataclass
class PathConfig:
    """All file paths used in evaluation."""
    # Input paths
    dataset_key: str = DEFAULT_DATASET
    image_root: Optional[Path] = None
    metadata_path: Optional[Path] = None

    imagenet_index_path: Path = Path("imagenet_class_index.txt")
    shapenet_mapping_path: Path = Path("ShapeNet-ImageNet1k-Mapping-Indexed-subcategories4.json")
    dataset_name: Optional[str] = None
    
    # Output paths
    results_dir: Path = Path("results")
    
    def __post_init__(self):
        dataset_spec = DATASETS[self.dataset_key]
        if self.image_root is None:
            self.image_root = dataset_spec.image_root
        if self.metadata_path is None:
            self.metadata_path = dataset_spec.metadata_path
        if self.dataset_name is None:
            self.dataset_name = self.image_root.name
    
    def get_model_output_dir(self, model_key: str) -> Path:
        """Get output directory for a specific model."""
        return self.results_dir / self.dataset_name / model_key
    
    def get_comparative_dir(self) -> Path:
        """Get directory for cross-model comparative results."""
        return self.results_dir / self.dataset_name / "comparative"

    def get_linear_probe_artifact_dir(self, model_key: str, reference_dataset_key: str) -> Path:
        """Get artifact directory for cached linear-probe features and heads."""
        return self.results_dir / "_artifacts" / "linear_probe" / model_key / reference_dataset_key
    
    def get_image_root(self) -> Path:
        """Get image root directory. Falls back to metadata parent if not set."""
        if self.image_root:
            return self.image_root
        return self.metadata_path.parent


# --- Evaluation Configuration ---

@dataclass
class EvalConfig:
    """Evaluation settings."""
    # General
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 4
    
    # Inference
    top_k: int = 5  # Store top-K predictions per image
    use_amp: bool = True  # Automatic mixed precision

    # Linear probing
    linear_probe_reference_dataset: str = "imagenet_1k"
    linear_probe_train_fraction: float = 0.8
    linear_probe_epochs: int = 3
    linear_probe_lr: float = 1e-2
    linear_probe_weight_decay: float = 1e-4
    linear_probe_batch_size: int = 4096
    
    # Determinism
    deterministic: bool = True
    benchmark: bool = False
    
    # Metrics
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    
    # Analysis
    stratify_by: List[str] = field(default_factory=lambda: [
        "Camera Position",
        "Fog",
        "Light Color (RGB)",
        "Material",
        "Object",
        "Level",
        "true_shapenet_superclass"
    ])


# --- Output Settings ---

@dataclass
class OutputConfig:
    """Output format settings."""
    # Formats
    save_predictions_csv: bool = True
    save_metrics_json: bool = True
    save_confusion_matrix: bool = True
    save_html_report: bool = True
    save_markdown_report: bool = True
    
    # CSV settings
    csv_separator: str = ";"
    
    # Visualization
    generate_plots: bool = True
    
    # Comparative analysis
    generate_comparative_report: bool = True


# --- Master Configuration ---

@dataclass
class Config:
    """Master configuration combining all settings."""
    paths: PathConfig = field(default_factory=PathConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Models to evaluate (defaults to all)
    models_to_evaluate: List[str] = field(default_factory=lambda: list(MODELS.keys()))
    
    def get_model_spec(self, model_key: str) -> ModelSpec:
        """Get ModelSpec for a given model key."""
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
        return MODELS[model_key]


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


# --- Metadata Schema ---

# Columns in Metadata_Expanded.csv
METADATA_COLUMNS = [
    "Image",
    "Object",
    "Level", 
    "Class",              # ImageNet index (0-999)
    "Material",
    "Camera Position",
    "Light Color (RGB)",
    "Fog",                # Boolean: true/false
    "ImageNet_Label",     # Human-readable ImageNet class name
    "ShapeNet_Superclass" # ShapeNet superclass or "<unmapped>"
]

# Columns used for stratified analysis
STRATIFICATION_COLUMNS = [
    "Camera Position",
    "Fog",
    "Light Color (RGB)", 
    "Material",
    "Object",
    "Level",
]

# Special value for unmapped classes
UNMAPPED_MARKER = "<unmapped>"
