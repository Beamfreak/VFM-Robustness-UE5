# Linear Probing Guide

This guide explains how to run the new linear-probe evaluation in the framework.

## What is new

The evaluator now supports a third variant type:

- `__logits` (native classifier head)
- `__knn` (feature extraction + kNN)
- `__linear_probe` (feature extraction + trained linear classifier)

Linear probing is currently enabled for:

- `dinov2_b_in1k`
- `clip_b_in1k`
- `swin_b_in1k`

## Important default behavior

By default, existing variant outputs are skipped if
`metrics_summary.json` already exists for that variant.

This means:

- rerunning `scripts.evaluate` will not recompute completed `__logits` or `__knn` variants
- it will only run missing variants (for example `__linear_probe` if not done yet)

To force a full rerun of all enabled variants, use:

- `--all-eval-types`

## 1) Install dependencies

Use the root requirements file:

```bash
pip install -r requirements.txt
```

## 2) (Optional) list models

```bash
python -m scripts.evaluate --list-models
```

## 3) Run linear probing for selected models

Recommended first run (on one dataset):

```bash
python -m scripts.evaluate \
  --dataset imagenet_9 \
  --models dinov2_b_in1k clip_b_in1k swin_b_in1k
```

What happens:

- for each selected model, variants are expanded
- completed variants are skipped by default
- linear-probe head is trained (or loaded from cache) and evaluated

## 4) Force full rerun when needed

If you want to rerun `__logits`, `__knn`, and `__linear_probe` even when outputs already exist:

```bash
python -m scripts.evaluate \
  --dataset imagenet_9 \
  --models dinov2_b_in1k clip_b_in1k swin_b_in1k \
  --all-eval-types
```

## 5) Rebuild comparative report only

If runs are already saved and you only want refreshed comparative outputs:

```bash
python -m scripts.evaluate --dataset imagenet_9 --rebuild-comparative
```

## 6) Aggregate across datasets

After running multiple datasets:

```bash
python -m scripts.aggregate_comparative_results
```

Outputs are generated under:

- `results/aggregate/KNN/`
- `results/aggregate/LOGITS/`
- `results/aggregate/LINEAR_PROBE/`

## Linear-probe artifacts and cache

Linear-probe training caches features and trained head under:

- `results/_artifacts/linear_probe/<model_key>/<reference_dataset_key>/`

Typical files:

- `train_features.npy`
- `train_labels.npy`
- `val_features.npy`
- `val_labels.npy`
- `probe_head.pt`

This avoids retraining/re-extracting features every run.

## Output locations for a dataset run

Per model variant:

- `results/<dataset_name>/<model_variant>/predictions.csv`
- `results/<dataset_name>/<model_variant>/metrics_summary.json`
- `results/<dataset_name>/<model_variant>/report.html`
- `results/<dataset_name>/<model_variant>/report.md`

Example variant folder:

- `results/imagenet-9/clip_b_in1k__linear_probe/`

## Suggested first execution sequence

1. Run one model on one dataset first:

```bash
python -m scripts.evaluate --dataset imagenet_9 --models clip_b_in1k
```

2. Inspect metrics/report files.
3. Run all three target models.
4. Rebuild comparative report if required.

## Troubleshooting

- If a run is skipped unexpectedly:
  - check whether `metrics_summary.json` already exists for that variant
  - rerun with `--all-eval-types` if needed

- If you hit GPU OOM:
  - lower batch size:

```bash
python -m scripts.evaluate --dataset imagenet_9 --models dinov2_b_in1k --batch-size 8
```

- If package install fails due to index/auth issues:
  - verify pip index points to public PyPI (`https://pypi.org/simple/`)

## Notes

- Linear-probe reference dataset is configured in `scripts/config.py` via:
  - `EvalConfig.linear_probe_reference_dataset`
- Train/val split fraction is controlled by:
  - `EvalConfig.linear_probe_train_fraction`
- Probe training hyperparameters are also defined in `EvalConfig`.

