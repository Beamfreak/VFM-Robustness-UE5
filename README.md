# Evaluating the Robustness of Vision Foundation Models (VFMs) Using Photorealistic Images Synthesized in Unreal Engine 5

This repository contains the evaluation framework, dataset metadata, and Unreal Engine 5 pipeline used to analyze the out-of-distribution (OOD) robustness of Vision Foundation Models (VFMs) as detailed in the thesis:

**"Evaluating the Robustness of Vision Foundation Models on an ImageNet Class-Subset Using Photorealistic Images Synthesized in Unreal Engine."**

## Overview

Modern vision models often fail under domain shift despite high in-distribution accuracy. This project provides a procedural synthetic rendering pipeline in Unreal Engine 5 to stress-test models like CLIP and DINOv2/v3 under controlled physical transformations.

By mapping ShapeNet 3D models to the ImageNet-1K taxonomy, we evaluate model performance at both fine-grained (leaf class) and semantic (superclass) levels.

## Key Datasets

We introduce two novel synthetic benchmarks with perfect causal granularity:

- **Multi-Factor Dataset:** Systematically varies 5 nuisance factors: Camera Viewpoint, Object Material, Illumination, Background Scene, and Atmospheric Fog.
- **Multi-Color Dataset:** Isolates texture bias by overriding object materials with uniform colored plastic across the primary spectrum.

## Features

- **UE5 Rendering Pipeline:** Built on an adapted Dataset Renderer plugin utilizing Lumen and Nanite for photorealism.
- **Triple Evaluation Strategy:** Standardized comparison across supervised and self-supervised architectures using Logits, k-NN, and Linear Probing.
- **Hierarchy-Aware Metrics:** Analysis of semantic vs. fine-grained robustness to distinguish between label confusion and total semantic collapse.
- **Advanced Diagnostics:** Includes Equivariance Analysis via transition vectors and Spatial Attribution using Grad-CAM and automated segmentation masks.

## Key Findings

- DINOv2/v3 models exhibit superior OOD robustness compared to language-supervised (CLIP) or standard supervised (ResNet) models.
- Top-down viewpoints represent a universal failure mode across all evaluated backbones.
- Supervised fine-tuning can destroy the latent invariance of pre-trained backbones, making linear probing a more reliable metric for OOD robustness.

---

## Repository Structure

```text
VFM-Robustness-UE5/
├── Datasets/
│   ├── Multi-Factor-Dataset/         # Metadata for the 5-factor synthetic benchmark
│   │   ├── Background/               # Per-factor image metadata (Background)
│   │   ├── Camera/                   # Per-factor image metadata (Camera viewpoint)
│   │   ├── Fog/                      # Per-factor image metadata (Atmospheric fog)
│   │   ├── Light/                    # Per-factor image metadata (Illumination)
│   │   ├── Material/                 # Per-factor image metadata (Object material)
│   │   ├── Metadata.csv              # Core image index with factor labels
│   │   └── Metadata_Expanded.csv     # Extended index with ImageNet label mappings
│   ├── Multi-Color-Dataset/          # Metadata for the texture-bias benchmark
│   │   ├── Camera/                   # Per-factor image metadata (Camera viewpoint)
│   │   ├── Material/                 # Per-factor image metadata (Color material)
│   │   ├── Metadata.csv              # Core image index with color/class labels
│   │   └── Metadata_Expanded.csv     # Extended index with ImageNet label mappings
│   ├── own_textures/                 # Custom PBR textures used in rendering
│   └── AssetList.txt             # Inventory of ShapeNet assets used
│
├── Experiments/
│   ├── scripts/                              # Full Python evaluation framework
│   │   ├── config.py                         # Model specs, dataset definitions & configuration
│   │   ├── evaluate.py                       # Primary CLI evaluation entrypoint
│   │   ├── models.py                         # Model loading via timm & HuggingFace
│   │   ├── data_loader.py                    # PyTorch Dataset & DataLoader utilities
│   │   ├── inference.py                      # Classifier head inference engine
│   │   ├── knn.py                            # Feature extraction & FAISS kNN evaluation
│   │   ├── linear_probe.py                   # Linear probe training & evaluation engine
│   │   ├── metrics.py                        # Classification & stratified metrics computation
│   │   ├── analysis.py                       # Stratified metadata factor analyzer
│   │   ├── reporting.py                      # HTML, Markdown, CSV & JSON report generators
│   │   ├── equivariance.py                   # Transition-vector equivariance analysis
│   │   ├── explain.py                        # Grad-CAM & segmentation-mask attribution
│   │   ├── aggregate_comparative_results.py  # Cross-dataset comparative aggregator
│   │   ├── preprocess_metadata.py            # Metadata expansion & label mapping
│   │   ├── preprocess_external_datasets.py   # External benchmark preprocessing
│   │   ├── preprocess_pug.py                 # PUG-ImageNet specific preprocessing
│   │   ├── run_all_evals.py                  # Batch evaluation runner (Python)
│   │   ├── run_all_evals.sh                  # Batch evaluation runner (Shell)
│   │   ├── csv_to_latex.py                   # CSV → LaTeX table converter
│   │   ├── generate_results_latex.py         # Results → LaTeX report generator
│   │   ├── plot_5_3_heatmap.py               # Factor × model heatmap plots
│   │   ├── plot_5_3_scatter.py               # Factor × model scatter plots
│   │   ├── plot_5_3_table.py                 # Factor × model table plots
│   │   ├── plot_5_6_curve_plot.py            # Accuracy-vs-factor curve plots
│   │   ├── utils.py                          # Shared utility helpers
│   │   ├── tools/
│   │   │   ├── calculate_averages.py         # Stratified factor average accuracy calculator
│   │   │   └── fix_csv_paths.py              # Relative path normalizer for metadata CSVs
│   │   └── PREPROCESSING_README.md           # Metadata preprocessing guide
│   ├── results/                              # Evaluated outputs (gitignored, generated at runtime)
│   │   ├── <dataset_name>/
│   │   │   ├── <model_variant>/              # Per-model CSVs, JSON, HTML & Markdown reports
│   │   │   └── comparative/                  # Per-dataset comparative rankings & report
│   │   ├── aggregate/                        # Cross-dataset aggregated summaries
│   │   │   └── {KNN,LOGITS,LINEAR_PROBE}/
│   │   └── _artifacts/                       # Cached linear probe features & classifier heads
│   ├── ShapeNet-ImageNet1k-Mapping-Indexed-subcategories4.json   # ShapeNet↔ImageNet taxonomy map
│   ├── imagenet_class_index.txt                                  # ImageNet-1K class index
│   ├── requirements.txt                                          # Python dependencies
│   ├── README.md                                                 # Evaluation framework documentation
│   └── LINEAR_PROBING_GUIDE.md                                   # Advanced linear probing guide
│
├── Unreal-Engine-Plugins/
│   ├── MultiFactorDatasetRenderer/             # UE5 plugin: 5-factor procedural renderer
│   ├── MultiColorDatasetRenderer/              # UE5 plugin: color-material procedural renderer
│   └── UE5 Dataset Generator Plugin Guide.pdf  # Plugin usage guide
│
├── LICENSE
└── README.md                                   # This file
```

---

## Quick Start

### 1. Evaluation

For full CLI reference and advanced usage, see [`Experiments/README.md`](Experiments/README.md).

### 2. Unreal Engine 5 Plugins

Install the UE5 plugins from `Unreal-Engine-Plugins/` into your UE5 project and follow the [`UE5 Dataset Generator Plugin Guide.pdf`](Unreal-Engine-Plugins/UE5%20Dataset%20Generator%20Plugin%20Guide.pdf) to configure and render new dataset images.

---

## Documentation

| Document | Description |
|---|---|
| [`Experiments/README.md`](Experiments/README.md) | Full evaluation framework reference (CLI, output structure, modes) |
| [`Experiments/LINEAR_PROBING_GUIDE.md`](Experiments/LINEAR_PROBING_GUIDE.md) | In-depth guide on linear probe training and feature caching |
| [`Experiments/scripts/PREPROCESSING_README.md`](Experiments/scripts/PREPROCESSING_README.md) | Metadata expansion and label mapping details |
| [`Unreal-Engine-Plugins/UE5 Dataset Generator Plugin Guide.pdf`](Unreal-Engine-Plugins/UE5%20Dataset%20Generator%20Plugin%20Guide.pdf) | UE5 plugin setup and rendering configuration |

---

## Citation

If you use this pipeline or dataset in your research, please cite:

```bibtex
@masterthesis{leinberger2026robustness,
  title={Evaluating the Robustness of Vision Foundation Models on an ImageNet Class-Subset Using Photorealistic Images Synthesized in Unreal Engine},
  author={Markus Leinberger},
  year={2026},
  school={University of Bamberg}
}
```
