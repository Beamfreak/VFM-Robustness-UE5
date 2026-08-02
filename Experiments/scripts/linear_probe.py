"""
Linear-probe evaluation for Vision Model Evaluation Framework.

Trains a linear classifier on top of frozen ImageNet-1k features and evaluates it
on the active dataset using the same output schema as logits/kNN evaluation.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import Config, DATASETS
from .data_loader import (
    build_reverse_mapping,
    create_dataloader,
    load_preprocessed_metadata,
    load_shapenet_mapping,
)
from .inference import InferenceEngine
from .knn import FeatureExtractor
from .models import ModelBundle
from .utils import ensure_dir, format_time, load_imagenet_index, optimize_dataframe_types, save_json


class FeatureMemmapDataset(Dataset):
    """Dataset wrapper around cached feature memmaps for linear-probe training."""

    def __init__(self, feature_path: Path, label_path: Path):
        self.features = np.load(feature_path, mmap_mode="r")
        self.labels = np.load(label_path)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(np.asarray(self.features[idx], dtype=np.float32))
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return features, label


class LinearProbeTrainer:
    """Train or load a cached linear probe on frozen ImageNet-1k features."""

    def __init__(
        self,
        bundle: ModelBundle,
        config: Config,
        feature_batch_size: Optional[int] = None,
    ):
        self.bundle = bundle
        self.config = config
        self.reference_dataset_key = config.eval.linear_probe_reference_dataset
        if self.reference_dataset_key not in DATASETS:
            raise ValueError(
                f"Unknown linear-probe reference dataset: {self.reference_dataset_key}. "
                f"Available: {list(DATASETS.keys())}"
            )

        self.reference_spec = DATASETS[self.reference_dataset_key]
        self.feature_batch_size = feature_batch_size or bundle.model_spec.batch_size
        self.artifact_dir = config.paths.get_linear_probe_artifact_dir(bundle.model_key, self.reference_dataset_key)
        ensure_dir(self.artifact_dir)

    def get_reference_split_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load reference metadata and deterministically split into train/val."""
        ref_df = load_preprocessed_metadata(self.reference_spec.metadata_path)
        train_idx, val_idx = stratified_train_val_split(
            labels=ref_df["Class"].to_numpy(),
            train_fraction=self.config.eval.linear_probe_train_fraction,
            seed=self.config.eval.seed,
        )

        train_df = ref_df.iloc[train_idx].reset_index(drop=True)
        val_df = ref_df.iloc[val_idx].reset_index(drop=True)

        if train_df.empty or val_df.empty:
            raise RuntimeError(
                "Linear-probe train/val split is empty. "
                "Check linear_probe_train_fraction and the reference metadata."
            )

        return train_df, val_df

    def load_or_train_probe(self, force_retrain: bool = False) -> nn.Module:
        """Return a trained linear head, reusing cached artifacts when possible."""
        checkpoint_path = self.artifact_dir / "probe_head.pt"

        if checkpoint_path.exists() and not force_retrain:
            return self._load_head(checkpoint_path)

        train_df, val_df = self.get_reference_split_metadata()
        train_cache = self._extract_split_features_to_cache("train", train_df)
        val_cache = self._extract_split_features_to_cache("val", val_df)
        return self._train_probe(train_cache, val_cache, checkpoint_path)

    def _extract_split_features_to_cache(self, split_name: str, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract and cache frozen backbone features for one split."""
        feature_path = self.artifact_dir / f"{split_name}_features.npy"
        label_path = self.artifact_dir / f"{split_name}_labels.npy"
        meta_path = self.artifact_dir / f"{split_name}_meta.json"

        if feature_path.exists() and label_path.exists() and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "feature_path": feature_path,
                "label_path": label_path,
                "n_samples": meta["n_samples"],
                "feature_dim": meta["feature_dim"],
                "num_classes": meta["num_classes"],
            }

        dataloader = create_dataloader(
            metadata_df=metadata_df,
            image_root=self.reference_spec.image_root,
            transform=self.bundle.transform,
            batch_size=self.feature_batch_size,
            num_workers=self.config.eval.num_workers,
            shuffle=False,
        )

        extractor = FeatureExtractor(self.bundle, use_amp=self.config.eval.use_amp)
        total = len(dataloader.dataset)
        feature_memmap = None
        labels = np.empty(total, dtype=np.int32)
        offset = 0

        for batch in tqdm(dataloader, desc=f"Linear-probe features ({split_name})"):
            features = extractor.extract_batch_features(batch).astype(np.float32, copy=False)
            if feature_memmap is None:
                feature_memmap = open_memmap(
                    feature_path,
                    mode="w+",
                    dtype="float32",
                    shape=(total, features.shape[1]),
                )

            batch_size = features.shape[0]
            feature_memmap[offset:offset + batch_size] = features
            labels[offset:offset + batch_size] = batch["true_imagenet_idx"].cpu().numpy().astype(np.int32)
            offset += batch_size

        if feature_memmap is None:
            raise RuntimeError(f"No features extracted for split '{split_name}'.")

        np.save(label_path, labels)
        meta = {
            "split": split_name,
            "n_samples": int(total),
            "feature_dim": int(feature_memmap.shape[1]),
            "num_classes": int(labels.max()) + 1,
        }
        save_json(meta, meta_path)

        return {
            "feature_path": feature_path,
            "label_path": label_path,
            "n_samples": meta["n_samples"],
            "feature_dim": meta["feature_dim"],
            "num_classes": meta["num_classes"],
        }

    def _train_probe(
        self,
        train_cache: Dict[str, Any],
        val_cache: Dict[str, Any],
        checkpoint_path: Path,
    ) -> nn.Module:
        """Train the linear probe from cached features and persist the best checkpoint."""
        device = self.bundle.device
        train_dataset = FeatureMemmapDataset(train_cache["feature_path"], train_cache["label_path"])
        val_dataset = FeatureMemmapDataset(val_cache["feature_path"], val_cache["label_path"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.eval.linear_probe_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval.linear_probe_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        head = nn.Linear(train_cache["feature_dim"], train_cache["num_classes"]).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=self.config.eval.linear_probe_lr,
            weight_decay=self.config.eval.linear_probe_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        best_state = None
        best_val_acc = -1.0
        start_time = time.time()

        for epoch in range(self.config.eval.linear_probe_epochs):
            head.train()
            running_loss = 0.0
            seen = 0

            for features, labels in tqdm(train_loader, desc=f"Linear probe train epoch {epoch + 1}", leave=False):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = head(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                batch_n = labels.size(0)
                running_loss += float(loss.item()) * batch_n
                seen += batch_n

            val_acc = self._evaluate_probe(head, val_loader)
            avg_loss = running_loss / max(1, seen)
            print(
                f"  Linear probe epoch {epoch + 1}/{self.config.eval.linear_probe_epochs}: "
                f"loss={avg_loss:.4f}, val_top1={val_acc:.4f}"
            )

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}

        if best_state is None:
            raise RuntimeError("Linear probe training did not produce a checkpoint.")

        torch.save(
            {
                "state_dict": best_state,
                "feature_dim": train_cache["feature_dim"],
                "num_classes": train_cache["num_classes"],
                "reference_dataset_key": self.reference_dataset_key,
                "train_fraction": self.config.eval.linear_probe_train_fraction,
                "best_val_top1": best_val_acc,
                "epochs": self.config.eval.linear_probe_epochs,
            },
            checkpoint_path,
        )
        print(f"  Linear probe trained in {format_time(time.time() - start_time)}")
        print(f"  Best validation Top-1: {best_val_acc:.4f}")

        return self._load_head(checkpoint_path)

    def _evaluate_probe(self, head: nn.Module, dataloader: DataLoader) -> float:
        """Evaluate the probe on cached validation features."""
        device = self.bundle.device
        head.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = head(features)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))

        return correct / total if total > 0 else 0.0

    def _load_head(self, checkpoint_path: Path) -> nn.Module:
        """Load a cached linear-probe head."""
        checkpoint = torch.load(checkpoint_path, map_location=self.bundle.device)
        head = nn.Linear(checkpoint["feature_dim"], checkpoint["num_classes"]).to(self.bundle.device)
        head.load_state_dict(checkpoint["state_dict"])
        head.eval()
        return head


def stratified_train_val_split(
    labels: np.ndarray,
    train_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic per-class train/val split without sklearn."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("linear_probe_train_fraction must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    labels = np.asarray(labels)
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)

        if class_indices.size == 1:
            train_indices.extend(class_indices.tolist())
            continue

        split_at = int(np.floor(class_indices.size * train_fraction))
        split_at = min(max(1, split_at), class_indices.size - 1)
        train_indices.extend(class_indices[:split_at].tolist())
        val_indices.extend(class_indices[split_at:].tolist())

    return np.array(sorted(train_indices)), np.array(sorted(val_indices))


def run_linear_probe_inference(
    bundle: ModelBundle,
    dataloader: DataLoader,
    config: Config,
    target_metadata_df: pd.DataFrame,
    force_retrain: bool = False,
) -> pd.DataFrame:
    """Train/load a linear probe and evaluate it on the active dataset."""
    print("\nRunning linear-probe evaluation...")
    start_time = time.time()

    trainer = LinearProbeTrainer(bundle=bundle, config=config, feature_batch_size=dataloader.batch_size)
    head = trainer.load_or_train_probe(force_retrain=force_retrain)

    # If the reference dataset itself is evaluated, use the held-out validation split.
    if config.paths.dataset_key == trainer.reference_dataset_key:
        _, target_metadata_df = trainer.get_reference_split_metadata()
        dataloader = create_dataloader(
            metadata_df=target_metadata_df,
            image_root=trainer.reference_spec.image_root,
            transform=bundle.transform,
            batch_size=dataloader.batch_size or bundle.model_spec.batch_size,
            num_workers=config.eval.num_workers,
            shuffle=False,
        )

    imagenet_index = load_imagenet_index(config.paths.imagenet_index_path)
    shapenet_mapping = load_shapenet_mapping(config.paths.shapenet_mapping_path)
    imagenet_to_shapenet = build_reverse_mapping(shapenet_mapping)
    engine = InferenceEngine(
        bundle=bundle,
        imagenet_index=imagenet_index,
        imagenet_to_shapenet=imagenet_to_shapenet,
        top_k=config.eval.top_k,
        use_amp=config.eval.use_amp,
    )
    extractor = FeatureExtractor(bundle, use_amp=config.eval.use_amp)

    results: List[Dict[str, Any]] = []
    head.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Linear probe inference ({bundle.model_spec.name})"):
            features = extractor.extract_batch_features_tensor(batch)
            logits = head(features)
            probs = F.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probs, config.eval.top_k, dim=1)

            topk_probs_np = topk_probs.cpu().numpy()
            topk_indices_np = topk_indices.cpu().numpy()

            for i in range(topk_indices_np.shape[0]):
                results.append(
                    engine._process_sample(
                        batch=batch,
                        sample_idx=i,
                        topk_indices=topk_indices_np[i],
                        topk_probs=topk_probs_np[i],
                    )
                )

    df = pd.DataFrame(results)
    df = optimize_dataframe_types(df)

    elapsed = time.time() - start_time
    print(f"  Linear-probe evaluation completed in {format_time(elapsed)}")
    print(f"  ImageNet Top-1 Accuracy: {df['imagenet_top1_correct'].mean():.1%}")

    return df


