# Vision Model Evaluation Framework (Eval-Framework)

A lightweight, robust, and extensible framework designed to evaluate vision foundation models (classifier heads, kNN feature probing, and linear-probe variants) across multiple synthetic and standard benchmarks with consistent metrics, factor stratification, and comparative reporting.

---

## Table of Contents

- [Features](#features)
- [Project Architecture](#project-architecture)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Evaluation Modes & Variants](#evaluation-modes--variants)
- [Output Structure](#output-structure)
- [Helper Tools](#helper-tools)
- [Detailed Guides](#detailed-guides)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Multiple Evaluation Modes**: Native classifier head (`__logits`), feature extraction + kNN (`__knn`), and frozen features + trained linear probing (`__linear_probe`).
- **Comprehensive Metrics**: Top-1/Top-5 accuracy, macro F1 score, confusion matrices, and 95% bootstrap confidence intervals.
- **Stratified Analysis**: Factor breakdowns across camera angles, lighting conditions, background materials, object levels, and superclasses.
- **Comparative Rankings**: Automated per-dataset model ranking CSVs, JSON summaries, and HTML visual reports.
- **Cross-Dataset Aggregations**: Consolidated benchmark master tables across all evaluated datasets.
- **Linear Probe Caching**: Feature and classifier head caching to avoid redundant feature extraction.

---

## Project Architecture

```text
Eval-Framework/
├── data/                                 # Dataset image roots & Metadata_Expanded.csv files
├── results/                              # Evaluated outputs & comparative summaries
│   ├── <dataset_name>/
│   │   ├── <model_variant>/              # Per-model variant results (CSV, JSON, HTML, MD)
│   │   └── comparative/                  # Per-dataset comparative rankings & report
│   ├── aggregate/                        # Cross-dataset aggregated summaries (KNN, LOGITS, LINEAR_PROBE)
│   └── _artifacts/                       # Cached linear probe features & heads
├── scripts/
│   ├── config.py                         # Model specs, dataset definitions, & configuration
│   ├── evaluate.py                       # Primary CLI evaluation entrypoint
│   ├── models.py                         # Model loading via timm & HuggingFace
│   ├── data_loader.py                    # PyTorch Dataset & DataLoader utilities
│   ├── inference.py                      # Classifier head inference engine
│   ├── knn.py                            # Feature extraction & FAISS kNN evaluation engine
│   ├── linear_probe.py                   # Linear probe training & evaluation engine
│   ├── metrics.py                        # Classification & stratified metrics computation
│   ├── analysis.py                       # Stratified metadata factor analyzer
│   ├── reporting.py                      # HTML, Markdown, CSV, & JSON report generators
│   ├── aggregate_comparative_results.py # Cross-dataset comparative aggregator
│   ├── preprocess_metadata.py            # Metadata expansion & label mapping
│   └── tools/                            # Utility scripts
│       ├── calculate_averages.py         # Stratified factor average accuracy calculator
│       └── fix_csv_paths.py              # Relative path normalizer for metadata CSVs
├── requirements.txt                      # Project dependencies
├── README.md                             # Master project documentation
└── LINEAR_PROBING_GUIDE.md               # Advanced linear probing guide
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Metadata (Required Once)

Before running evaluations, generate `Metadata_Expanded.csv` for your target dataset:

```bash
python scripts/preprocess_metadata.py
```

### 3. List Available Models

```bash
python -m scripts.evaluate --list-models
```

### 4. Run Model Evaluation

Evaluate all models on the default dataset:

```bash
python -m scripts.evaluate
```

Evaluate selected models on a specific dataset:

```bash
python -m scripts.evaluate --dataset imagenet_9 --models clip_b resnet50_in1k
```

---

## CLI Reference

### `scripts.evaluate` Options

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--dataset` | `-d` | Target dataset key (defined in `scripts/config.py`). | `normal` |
| `--models` | `-m` | List of model keys to evaluate. | All configured models |
| `--device` | | Execution device (`cuda` or `cpu`). | `cuda` |
| `--batch-size` | `-b` | Override default batch size per model. | Model spec default |
| `--skip-inference` | | Skip feature extraction/inference if `predictions.csv` exists. | `False` |
| `--all-eval-types` | | Force rerun all enabled variants even if outputs exist. | `False` |
| `--rebuild-comparative` | | Rebuild comparative summary report from existing runs without rerunning inference. | `False` |
| `--list-models` | | Print available models and exit. | `False` |
| `--metadata` | | Custom path to `Metadata_Expanded.csv`. | Dataset spec default |
| `--output-dir` | `-o` | Custom results output directory. | `results/` |

### Command Examples

**Run with reduced batch size (for large models/kNN memory pressure):**
```bash
python -m scripts.evaluate --models dinov2_l --batch-size 8
```

**Rebuild comparative report only (without recompute):**
```bash
python -m scripts.evaluate --dataset imagenet_9 --rebuild-comparative
```

**Force rerun all evaluation types (`__logits`, `__knn`, `__linear_probe`):**
```bash
python -m scripts.evaluate --dataset imagenet_9 --models clip_b_in1k --all-eval-types
```

**Aggregate comparative reports across all evaluated datasets:**
```bash
python -m scripts.aggregate_comparative_results
```

---

## Evaluation Modes & Variants

`scripts.evaluate` expands requested base model keys into evaluation variants:

- `__logits`: Evaluates native ImageNet classifier heads.
- `__knn`: Extracts feature representations and evaluates via k-Nearest Neighbors.
- `__linear_probe`: Trains a linear classifier on frozen model features.

By default, existing variant runs with `metrics_summary.json` are skipped unless `--all-eval-types` is specified.

---

## Output Structure

### Per Model Variant Run
`results/<dataset_name>/<model_variant>/`
- `predictions.csv`: Detailed image-level predictions and ground truth.
- `metrics_summary.json`: Top-1/Top-5 accuracy, macro F1, and bootstrap CIs.
- `metrics_per_class_imagenet.csv`: Per-class ImageNet accuracy.
- `stratified_*.csv`: Accuracy breakdowns across metadata factors.
- `report.html` / `report.md`: Visual and textual model summary reports.

### Per Dataset Comparative
`results/<dataset_name>/comparative/`
- `model_rankings.csv`: Comparative metric rankings across models.
- `comparative_summary.json`: Structured benchmark metrics per model.
- `comparative_report.html`: Visual comparative report.

### Cross-Dataset Aggregates
`results/aggregate/{KNN, LOGITS, LINEAR_PROBE}/`
- `01_master_all_models_all_datasets.csv`: Master benchmark table.
- `02_by_dataset/`: Per-dataset summary CSVs.
- `03_by_metric/`: Per-metric summary CSVs.
- `04_by_model/`: Per-model summary CSVs.
- `summary.html`: Interactive aggregate report.

---

## Helper Tools

Located under `scripts/tools/`:

### Calculate Stratified Factor Averages
Computes average Top-1 accuracy across metadata factor levels (excluding `"Default"` values):
```bash
python scripts/tools/calculate_averages.py --dataset-results-dir results/PUG_ImageNet
```

### Fix CSV Path Prefixes
Normalizes image relative paths in `Metadata_Expanded.csv` files:
```bash
python scripts/tools/fix_csv_paths.py --data-dir data
```

---

## Detailed Guides

- [Linear Probing Guide](LINEAR_PROBING_GUIDE.md): In-depth guide on linear probe training, hyperparameters, and feature caching.
- [Metadata Preprocessing Guide](scripts/PREPROCESSING_README.md): Details on metadata expansion and label mapping.

---

## Troubleshooting

- **Missing Metadata Error**: Run `python scripts/preprocess_metadata.py` to create `Metadata_Expanded.csv`.
- **CUDA Out-of-Memory (OOM)**: Override batch size to a smaller value using `--batch-size 8`.
- **Skipped Model Runs**: Evaluation skips completed variants containing `metrics_summary.json`. Use `--all-eval-types` to force rerunning.
