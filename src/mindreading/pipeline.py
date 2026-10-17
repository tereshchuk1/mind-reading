"""End-to-end training and inference.

    python -m mindreading.pipeline train   --config configs/final.json
    python -m mindreading.pipeline predict --config configs/final.json

train   : fits every ensemble member on the TRAINING recordings of all six
          subjects (group-average target) and stores weights in weights/.
predict : predicts the EEG response of the 102 candidate videos, ranks them for
          every test recording in the ORIGINAL file order, writes
          submission/sub-0X_ranked_candidate_list.npy, and scores the identical
          code path on the dev subjects (sub-01..04) as a format/sanity check.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np

from .data import ALL_SUBJECTS, DEV_SUBJECTS, HELDOUT_SUBJECTS, TEST_IDS, load_subject, load_test_labels, path
from .model import TRF, FeatureSpace, col_zscore, group_average, load_feature, rank_scores, ranked_ids, similarity


def _feats(names, split):
    return {n: load_feature(n)[0 if split == "train" else 1] for n in names}


def train(cfg):
    prep = cfg["prep"]
    Y, cnt = group_average(ALL_SUBJECTS, **prep)
    m = cnt > 0
    os.makedirs(path("weights"), exist_ok=True)
    for i, mem in enumerate(cfg["members"]):
        fs = FeatureSpace(mem["features"], mem.get("n_pca", 30), Y.shape[-1]).fit(_feats(mem["features"], "train"))
        F = fs.transform(_feats(mem["features"], "train"))
        trf = TRF(lags=range(*mem.get("lags", [0, 30, 2])), seed=mem.get("seed", 0)).fit(F[m], Y[m], w=cnt[m])
        out = path("weights", f"member{i}.joblib")
        joblib.dump({"member": mem, "prep": prep, "space": fs, "W": trf.W, "lags": trf.lags,
                     "alpha": trf.alpha_, "cv_score": float(-trf.cv_err_.min() / trf.k)}, out, compress=3)
        print(f"member {i} {'+'.join(mem['features'])}: alpha={trf.alpha_:.0f} -> {out}", flush=True)


def predict_candidates(cfg):
    """Predicted EEG (102,C,T) of each candidate test video, per ensemble member."""
    preds = []
    for i in range(len(cfg["members"])):
        b = joblib.load(path("weights", f"member{i}.joblib"))
        trf = TRF(lags=b["lags"])
        trf.W = b["W"]
        preds.append(trf.predict(b["space"].transform(_feats(b["member"]["features"], "test"))))
    return preds


def score_matrix(cfg, T, preds):
    """Fused (recordings x candidates) score; members are averaged after the
    unsupervised column z-scoring, weights from cfg (default equal)."""
    w = cfg.get("weights", [1.0] * len(preds))
    return sum(wi * col_zscore(similarity(T, P)) for wi, P in zip(w, preds)) / sum(w)


def predict(cfg, out_dir="submission"):
    prep = cfg["prep"]
    preds = predict_candidates(cfg)
    os.makedirs(path(out_dir), exist_ok=True)
    report = {}
    for s in ALL_SUBJECTS:
        _, _, T = load_subject(s, **prep)          # original (shuffled) order, never re-sorted
        ranked = ranked_ids(score_matrix(cfg, T, preds), TEST_IDS)
        assert ranked.shape == (102, 102) and all(sorted(r) == list(TEST_IDS) for r in ranked)
        if s in HELDOUT_SUBJECTS:
            f = path(out_dir, f"sub-{s:02d}_ranked_candidate_list.npy")
            np.save(f, ranked)
            print("wrote", f)
        else:
            gt = load_test_labels(s)
            top1 = float((ranked[:, 0] == gt).mean())
            top5 = float((ranked[:, :5] == gt[:, None]).any(1).mean())
            report[f"sub-{s:02d}"] = {"top1": top1, "top5": top5}
            print(f"sub-{s:02d}  top-1 {top1:.4f}  top-5 {top5:.4f}")
    report["mean_dev"] = {k: float(np.mean([v[k] for v in report.values()])) for k in ("top1", "top5")}
    print("dev mean", report["mean_dev"])
    with open(path(out_dir, "dev_scores.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "predict", "all"])
    ap.add_argument("--config", default=path("configs", "final.json"))
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.stage in ("train", "all"):
        train(cfg)
    if a.stage in ("predict", "all"):
        predict(cfg)


if __name__ == "__main__":
    main()
