"""Analyses and figures for the report (run after `pipeline train`).

    python -m mindreading.analysis

figures/trf_kernels.png      - TRF weight energy per lag for every feature set
                               (the latency at which each stimulus property
                               drives the EEG) and its channel topography.
figures/time_windows.png     - identification accuracy from 0.5 s sliding windows.
figures/failure_audio.png    - rank of the true video vs. audio loudness.
figures/ceiling.json         - retrieval with "perfect" templates (other dev
                               subjects' test responses; analysis only, never
                               used for prediction).
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import DEV_SUBJECTS, load_subject, load_test_labels, path
from .model import col_zscore, load_feature, rank_scores, similarity
from .pipeline import predict_candidates

FIG = path("figures")


def kernels(cfg):
    b = joblib.load(path("weights", "member0.joblib"))
    W, lags, fs = b["W"], b["lags"], b["space"]
    D = sum(p.n_components_ for _, p in fs.tf.values())
    Wl = W[: D * len(lags)].reshape(len(lags), D, -1)           # (lag, feature, channel)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    start = 0
    dt = 1000 * b["prep"]["dec"] / 500
    for n in fs.names:
        k = fs.tf[n][1].n_components_
        e = (Wl[:, start:start + k] ** 2).sum((1, 2))
        ax[0].plot(np.array(lags) * dt, e / e.max(), label=n, marker="o", ms=3)
        topo = (Wl[:, start:start + k] ** 2).sum((0, 1))
        ax[1].plot(np.arange(1, 18), topo / topo.max(), label=n, marker="o", ms=3)
        start += k
    ax[0].set(xlabel="lag (ms)", ylabel="normalised kernel energy", title="When each stimulus property drives EEG")
    ax[1].set(xlabel="EEG channel index", ylabel="normalised kernel energy", title="Where (channel profile)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "trf_kernels.png"), dpi=130)


def time_windows(cfg, preds):
    prep = cfg["prep"]
    T0 = preds[0].shape[-1]
    sr = 500 / prep["dec"]
    w = int(0.5 * sr)
    starts = np.arange(0, T0 - w + 1, w // 2)
    res = []
    for st in starts:
        win = slice(st, st + w)
        t1 = []
        for s in DEV_SUBJECTS:
            _, _, T = load_subject(s, **prep)
            S = sum(col_zscore(similarity(T, P, win)) for P in preds)
            t1.append(rank_scores(S, load_test_labels(s) - 1001)["top5"])
        res.append(np.mean(t1) * 100)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot((starts + w / 2) / sr, res, marker="o")
    ax.axhline(5 / 102 * 100, ls="--", c="gray", label="chance")
    ax.set(xlabel="window centre (s), 0.5 s windows", ylabel="dev top-5 (%)", title="Information over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "time_windows.png"), dpi=130)
    return dict(zip(np.round((starts + w / 2) / sr, 2).tolist(), np.round(res, 1).tolist()))


def failure_audio(cfg, preds):
    _, te = load_feature("audio50")
    loud = te[:, :, 32].mean(1)                                   # broadband log-envelope
    ranks = []
    for s in DEV_SUBJECTS:
        _, _, T = load_subject(s, **cfg["prep"])
        S = sum(col_zscore(similarity(T, P)) for P in preds)
        lab = load_test_labels(s) - 1001
        ranks.append(((-S).argsort(1).argsort(1)[np.arange(102), lab] + 1, lab))
    r = np.zeros(102)
    for rk, lab in ranks:
        r[lab] += rk / len(ranks)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(loud, r, s=12)
    ax.set(xlabel="mean log loudness of the test clip", ylabel="mean rank of true video (dev)", title="Failure cases: quiet clips")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "failure_audio.png"), dpi=130)
    q = loud < np.quantile(loud, 0.25)
    return {"rank_quietest_quarter": float(r[q].mean()), "rank_rest": float(r[~q].mean()),
            "spearman_loudness_vs_rank": float(np.corrcoef(np.argsort(np.argsort(loud)), np.argsort(np.argsort(r)))[0, 1]),
            "worst_videos": (1001 + np.argsort(-r)[:10]).tolist(), "best_videos": (1001 + np.argsort(r)[:10]).tolist()}


def ceiling(prep):
    out = {}
    for s in DEV_SUBJECTS:
        _, _, T = load_subject(s, **prep)
        Ts = T[np.argsort(load_test_labels(s))]
        tm = np.mean([load_subject(o, **prep)[2][np.argsort(load_test_labels(o))] for o in DEV_SUBJECTS if o != s], 0)
        out[f"sub-{s:02d}"] = rank_scores(col_zscore(similarity(Ts, tm)), np.arange(102))
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    cfg = json.load(open(path("configs", "final.json")))
    preds = predict_candidates(cfg)
    kernels(cfg)
    res = {"time_windows_top5": time_windows(cfg, preds), "failure_audio": failure_audio(cfg, preds),
           "ceiling_other_subjects_templates": ceiling(cfg["prep"])}
    with open(os.path.join(FIG, "analysis.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
