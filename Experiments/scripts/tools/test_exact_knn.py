#!/usr/bin/env python3
"""
Test Exact vs. IVF k-NN Search Performance.

Benchmarks exact FAISS inner-product search against approximate IVF search
on feature extractions to verify indexing accuracy and performance trade-offs.
"""

import sys
import time
from pathlib import Path
import numpy as np
import faiss

# Ensure repository root / scripts directory is in sys.path
sys_path_root = Path(__file__).resolve().parent.parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from scripts.config import get_default_config
from scripts.models import ModelLoader
from scripts.data_loader import load_preprocessed_metadata, create_dataloader
from scripts.knn import FeatureExtractor


def run_test():
    config = get_default_config()
    config.paths.dataset_key = 'imagenet_1k'
    config.paths.__post_init__()

    # Load metadata and take a small shard
    meta = load_preprocessed_metadata(config.paths.metadata_path)
    N = 2000
    meta_shard = meta.iloc[:N].reset_index(drop=True)

    results = {}
    for model_key in ['dinov1_b', 'dinov3_b']:
        print('---', model_key, '---')
        loader = ModelLoader(config.eval.device)
        bundle = loader.load_model(model_key, override_eval_mode='knn')

        dl = create_dataloader(
            metadata_df=meta_shard,
            image_root=config.paths.get_image_root(),
            transform=bundle.transform,
            batch_size=64,
            num_workers=2,
            shuffle=False
        )

        ext = FeatureExtractor(bundle, use_amp=False)
        t0 = time.time()
        X, meta_out = ext.extract_features(dl)
        t1 = time.time()
        print('Extracted', X.shape, 'in', round(t1 - t0, 1), 's')

        labels = np.array(meta_out['true_imagenet_idx'])

        # Exact index
        dim = X.shape[1]
        idx_exact = faiss.IndexFlatIP(dim)
        idx_exact.add(X)

        # IVF index with low nprobe (simulates original approx)
        n_samples = X.shape[0]
        nlist = min(4096, max(64, int(np.sqrt(n_samples))))
        nlist = min(nlist, max(1, n_samples // 40))
        quantizer = faiss.IndexFlatIP(dim)
        idx_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        train_size = min(n_samples, max(200000, nlist * 50))
        train_feats = X[:train_size]
        idx_ivf.train(train_feats)
        idx_ivf.add(X)
        idx_ivf.nprobe = min(64, max(8, nlist // 16))

        # Query with leave-one-out
        k = 20
        fetch_k = k + 1

        def predict_from_index(index, Xq):
            sims, inds = index.search(Xq, fetch_k)
            preds = []
            for i in range(Xq.shape[0]):
                found = -1
                for j in range(fetch_k):
                    ni = inds[i, j]
                    if ni == -1:
                        continue
                    if ni == i:  # leave-one-out
                        continue
                    found = labels[ni]
                    break
                preds.append(found)
            return np.array(preds)

        pred_exact = predict_from_index(idx_exact, X)
        acc_exact = (pred_exact == labels).mean()
        print('Exact kNN top1:', round(float(acc_exact), 4))

        pred_ivf = predict_from_index(idx_ivf, X)
        acc_ivf = (pred_ivf == labels).mean()
        print('IVF low-nprobe top1:', round(float(acc_ivf), 4))

        idx_ivf.nprobe = min(256, max(32, nlist // 4))
        pred_ivf_hi = predict_from_index(idx_ivf, X)
        acc_ivf_hi = (pred_ivf_hi == labels).mean()
        print('IVF high-nprobe top1:', round(float(acc_ivf_hi), 4))

        results[model_key] = (acc_exact, acc_ivf, acc_ivf_hi)

    print('\nSummary:')
    for k, v in results.items():
        print(k, 'exact,ivf_low,ivf_hi =', tuple(round(float(x), 4) for x in v))


if __name__ == '__main__':
    run_test()
