"""
Vision Model Evaluation Framework

A framework for evaluating vision models on ImageNet classification
and ShapeNet superclass prediction tasks.

Modules:
    - config: Configuration classes and model definitions
    - utils: Utility functions (seeds, file I/O, metrics helpers)
    - data_loader: Dataset and DataLoader utilities
    - models: Model loading via timm
    - inference: Batch inference engine
    - metrics: Metrics computation
    - analysis: Stratified analysis
    - reporting: Report generation (HTML, JSON, CSV, Markdown)
    - evaluate: Main evaluation entry point

Usage:
    # From command line
    python -m scripts.evaluate --models clip_b resnet50
    
    # From Python
    from scripts.evaluate import Evaluator
    from scripts.config import get_default_config
    
    config = get_default_config()
    evaluator = Evaluator(config)
    results = evaluator.run(models=["clip_b", "resnet50"])
"""

from .config import (
    Config,
    ModelSpec,
    MODELS,
    BASELINE_MODEL,
    get_default_config,
    PathConfig,
    EvalConfig,
    OutputConfig
)

from .utils import (
    set_seed,
    set_deterministic,
    load_imagenet_index,
)

from .data_loader import (
    load_preprocessed_metadata,
    create_dataloader,
    get_dataset_statistics,
    EvaluationDataset
)

from .models import (
    ModelLoader,
    ModelBundle,
    get_available_models
)

from .inference import (
    InferenceEngine,
    run_model_inference,
    save_predictions,
    load_predictions
)

from .metrics import (
    MetricsComputer,
    compute_metrics_from_predictions
)

from .analysis import (
    StratifiedAnalyzer,
    run_stratified_analysis
)

from .reporting import (
    ReportGenerator,
    ComparativeReportGenerator,
    generate_model_report,
    generate_comparative_report
)

# Note: Evaluator is imported lazily to avoid circular import issues
# when running `python -m scripts.evaluate`
def __getattr__(name):
    if name == "Evaluator":
        from .evaluate import Evaluator
        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.0.0"
__author__ = "Vision Model Evaluation Framework"

__all__ = [
    # Config
    "Config",
    "ModelSpec",
    "MODELS",
    "BASELINE_MODEL",
    "get_default_config",
    "PathConfig",
    "EvalConfig",
    "OutputConfig",
    # Utils
    "set_seed",
    "set_deterministic",
    "load_imagenet_index",
    # Data
    "load_preprocessed_metadata",
    "create_dataloader",
    "get_dataset_statistics",
    "EvaluationDataset",
    # Models
    "ModelLoader",
    "ModelBundle",
    "get_available_models",
    # Inference
    "InferenceEngine",
    "run_model_inference",
    "save_predictions",
    "load_predictions",
    # Metrics
    "MetricsComputer",
    "compute_metrics_from_predictions",
    # Analysis
    "StratifiedAnalyzer",
    "run_stratified_analysis",
    # Reporting
    "ReportGenerator",
    "ComparativeReportGenerator",
    "generate_model_report",
    "generate_comparative_report",
    # Main
    "Evaluator",
]
