"""End-to-end training and inference.

    python -m mindreading.pipeline train   --config configs/final.json
    python -m mindreading.pipeline predict --config configs/final.json
    python -m mindreading.pipeline all --config configs/v2.json --out-weights weights/v2 --out-submission submission/v2

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


def train(cfg, out_weights="weights"):
    prep = cfg["prep"]
    Y, cnt = group_average(ALL_SUBJECTS, **prep)
    m = cnt > 0
    os.makedirs(path(out_weights), exist_ok=True)
    for i, mem in enumerate(cfg["members"]):
        # "band": per-feature-set scales = banded ridge (see banded.py), stored in the space
        fs = FeatureSpace(mem["features"], mem.get("n_pca", 30), Y.shape[-1], scales=mem.get("band")).fit(_feats(mem["features"], "train"))
        F = fs.transform(_feats(mem["features"], "train"))
        trf = make_trf(mem).fit(F[m], Y[m], w=cnt[m])
        out = path(out_weights, f"member{i}.joblib")
        joblib.dump({"member": mem, "prep": prep, "space": fs, "W": trf.W, "lags": trf.lags,
                     "alpha": trf.alpha_, "cv_score": float(-trf.cv_err_.min() / trf.k)}, out, compress=3)
        print(f"member {i} {'+'.join(mem['features'])}: alpha={trf.alpha_:.0f} -> {out}", flush=True)
    if cfg.get("denoise"):
        train_denoiser(cfg, Y, m, out_weights)


def train_denoiser(cfg, Y, m, out_weights):
    """CCA between the ensemble's out-of-fold predictions (10-fold over TRAINING
    videos) and the group-average training EEG; keep the first k canonical
    components and back-project them to the channels (v2, see cca.py)."""
    from .cca import cca_filters, denoiser
    from .evaluate import member_predictions

    dn = cfg["denoise"]
    oofs = [member_predictions(mem["features"], mem.get("n_pca", 30), tuple(mem.get("lags", [0, 30, 2])),
                               cfg["prep"], band=mem.get("band"))[1] for mem in cfg["members"]]
    Wx, Wy, rho, Rx, Ry = cca_filters(np.mean(oofs, 0)[m], Y[m], dn.get("reg", 0.1))
    out = path(out_weights, "denoise.joblib")
    joblib.dump({"Dx": denoiser(Wx, Rx, dn["k"]), "Dy": denoiser(Wy, Ry, dn["k"]), "rho": rho, "cfg": dn}, out)
    print(f"denoiser CCA k={dn['k']} canonical r={np.round(rho[:4], 3).tolist()} -> {out}", flush=True)


def make_trf(mem):
    return TRF(lags=range(*mem.get("lags", [0, 30, 2])), seed=mem.get("seed", 0))


def predict_candidates(cfg, weights_dir="weights"):
    """Predicted EEG (102,C,T) of each candidate test video, per ensemble member."""
    preds = []
    for i in range(len(cfg["members"])):
        b = joblib.load(path(weights_dir, f"member{i}.joblib"))
        trf = make_trf(b["member"])
        trf.W = b["W"]
        preds.append(trf.predict(b["space"].transform(_feats(b["member"]["features"], "test"))))
    return preds


def score_matrix(cfg, T, preds, dn=None):
    """Fused (recordings x candidates) score; members are averaged after the
    unsupervised column z-scoring, weights from cfg (default equal). dn: optional
    spatial denoiser {"Dx", "Dy"} applied to predictions / recordings."""
    w = cfg.get("weights", [1.0] * len(preds))
    if dn is not None:
        T = np.einsum("nct,cd->ndt", T, dn["Dy"])
        preds = [np.einsum("nct,cd->ndt", P, dn["Dx"]) for P in preds]
    return sum(wi * col_zscore(similarity(T, P)) for wi, P in zip(w, preds)) / sum(w)


def predict(cfg, out_dir="submission", weights_dir="weights"):
    prep = cfg["prep"]
    preds = predict_candidates(cfg, weights_dir)
    dn = joblib.load(path(weights_dir, "denoise.joblib")) if cfg.get("denoise") else None
    os.makedirs(path(out_dir), exist_ok=True)
    report = {}
    for s in ALL_SUBJECTS:
        _, _, T = load_subject(s, **prep)          # original (shuffled) order, never re-sorted
        ranked = ranked_ids(score_matrix(cfg, T, preds, dn), TEST_IDS)
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
    ap.add_argument("--out-weights", default="weights", help="weights directory (relative to the repo root)")
    ap.add_argument("--out-submission", default="submission", help="submission directory (relative to the repo root)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.stage in ("train", "all"):
        train(cfg, a.out_weights)
    if a.stage in ("predict", "all"):
        predict(cfg, a.out_submission, a.out_weights)


if __name__ == "__main__":
    main()
