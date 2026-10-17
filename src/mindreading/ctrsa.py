"""Cross-temporal representational similarity analysis (CT-RSA), TRAINING data only.
Analysis for the report; it has no effect on the submission.

    python -m mindreading.ctrsa [--videos 600] [--perms 500]

Neural RDMs: for every 60 ms EEG window (3 samples at 50 Hz) and channel group,
patterns = channels x samples of the split-half group averages (subjects 1-3 vs
4-6). Cross-validated correlation distance d(i,j) = 1 - (r(A_i,B_j) + r(A_j,B_i))/2
(noise is independent between halves, so it does not inflate distances).
Model RDMs: per layer and per model time bin, correlation distance between the
videos' feature vectors (features z-scored over videos).
Cross-temporal matrix: Spearman(neural RDM at EEG time t, model RDM at model time tau).
Permutation test: videos of the model RDM are shuffled (same permutations for all
cells and both channel groups); max-statistic over the matrix gives a family-wise
threshold; the lateral-minus-occipital difference of the mean over causal cells
(0 <= t - tau <= 600 ms) gets its own permutation p-value.
Noise ceiling per EEG window: split-half reliability r of the within-half RDMs,
Spearman-Brown corrected 2r/(1+r).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from .data import LATERAL_PARIETAL, OCCIPITAL, chan_idx, load_subject, path

GROUPS = {"occipital": OCCIPITAL, "lateral_parietal": LATERAL_PARIETAL}
EEG_POOL = 3                       # samples at 50 Hz -> 60 ms windows
FIG = path("figures")


def split_half(prep=dict(lowpass=20.0, dec=10)):
    """Mean training response per video for subjects 1-3 (A) and 4-6 (B)."""
    halves = []
    for subs in ((1, 2, 3), (4, 5, 6)):
        acc, cnt = np.zeros((1000, 17, 150)), np.zeros(1000)
        for s in subs:
            ids, X, _ = load_subject(s, **prep)
            acc[ids - 1] += X
            cnt[ids - 1] += 1
        halves.append((acc / np.maximum(cnt, 1)[:, None, None], cnt))
    (A, ca), (B, cb) = halves
    ok = (ca > 0) & (cb > 0)
    return A, B, ok


def _rows(P):
    """Standardise rows (videos x dims) for correlation."""
    P = P - P.mean(1, keepdims=True)
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)


def _triu(M):
    return M[np.triu_indices(len(M), 1)]


def neural_rdms(A, B, chans):
    """-> cross-validated RDMs (n_win, n_pairs), and within-half RDMs for the ceiling."""
    n, _, T = A.shape
    nw = T // EEG_POOL
    cv, ra, rb = [], [], []
    for w in range(nw):
        sl = slice(w * EEG_POOL, (w + 1) * EEG_POOL)
        a = _rows(A[:, chans, sl].reshape(n, -1))
        b = _rows(B[:, chans, sl].reshape(n, -1))
        r = a @ b.T
        cv.append(_triu(1 - (r + r.T) / 2))
        ra.append(_triu(1 - a @ a.T))
        rb.append(_triu(1 - b @ b.T))
    return np.array(cv), np.array(ra), np.array(rb)


def model_rdms(F):
    """F: (n_videos, n_time, D) -> (n_time, n_pairs)."""
    out = []
    for t in range(F.shape[1]):
        X = F[:, t].astype(np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        Z = _rows(X)
        out.append(_triu(1 - Z @ Z.T))
    return np.array(out)


def _rank_z(R):
    """Rank-transform each row and standardise -> Spearman = mean product."""
    Rk = np.apply_along_axis(rankdata, 1, R).astype(np.float32)
    Rk -= Rk.mean(1, keepdims=True)
    return Rk / Rk.std(1, keepdims=True)


def pool_time(F, k):
    n, T, D = F.shape
    T2 = T // k
    return F[:, : T2 * k].reshape(n, T2, k, D).mean(2)


def model_layers():
    """{model: {layer: (features (1000,T,D), seconds per bin)}} from cache/ (train only)."""
    out = {}
    w = {}
    for l in range(7):
        f = path("cache", f"whisper_L{l}_train.npy")
        if os.path.exists(f):
            w[f"L{l}"] = (pool_time(np.load(f).astype(np.float32), 3), 0.12)     # 25 Hz -> 120 ms bins
    if w:
        out["whisper"] = w
    a = {}
    for c in range(1, 6):
        f = path("cache", f"alexnet_c{c}_train.npy")
        if os.path.exists(f):
            x = np.load(f).astype(np.float32)
            a[f"conv{c}"] = (x, 3.0 / x.shape[1])
    if a:
        out["alexnet"] = a
    if os.path.exists(path("cache", "vjepa2_train.npy")):
        v = np.load(path("cache", "vjepa2_train.npy"))
        h = v.shape[-1] // 2
        out["vjepa2"] = {"mid": (pool_time(v[..., :h], 2), 0.1875), "last": (pool_time(v[..., h:], 2), 0.1875)}
    return out


def run(n_videos=600, n_perm=500, seed=0):
    A, B, ok = split_half()
    rng = np.random.default_rng(seed)
    vids = np.sort(rng.choice(np.where(ok)[0], size=min(n_videos, ok.sum()), replace=False))
    A, B = A[vids], B[vids]
    n = len(vids)
    eeg_dt = EEG_POOL / 50
    iu = np.triu_indices(n, 1)
    pair_index = np.zeros((n, n), np.int64)
    pair_index[iu] = np.arange(len(iu[0]))
    pair_index = pair_index + pair_index.T
    perms = [rng.permutation(n) for _ in range(n_perm)]
    perm_idx = [pair_index[p[iu[0]], p[iu[1]]] for p in perms]

    neural, ceiling = {}, {}
    for g, ch in GROUPS.items():
        cv, ra, rb = neural_rdms(A, B, chan_idx(ch))
        neural[g] = _rank_z(cv)
        rel = (_rank_z(ra) * _rank_z(rb)).mean(1)
        ceiling[g] = np.round(2 * rel / (1 + rel), 4).tolist()
    n_win = len(next(iter(neural.values())))
    t_eeg = (np.arange(n_win) + 0.5) * eeg_dt

    res = {"n_videos": int(n), "n_perm": n_perm, "eeg_window_s": eeg_dt, "noise_ceiling_sb": ceiling, "models": {}}
    for model, layers in model_layers().items():
        res["models"][model] = {}
        fig, axes = plt.subplots(len(GROUPS), len(layers), figsize=(2.6 * len(layers), 5.2), squeeze=False)
        for j, (lname, (F, mdt)) in enumerate(layers.items()):
            M = _rank_z(model_rdms(F[vids]))
            t_mod = (np.arange(len(M)) + 0.5) * mdt
            causal = ((t_eeg[:, None] - t_mod[None]) >= 0) & ((t_eeg[:, None] - t_mod[None]) <= 0.6)
            obs = {g: neural[g] @ M.T / M.shape[1] for g in GROUPS}         # (eeg time, model time)
            null_max = {g: [] for g in GROUPS}
            null_diff = []
            for pi in perm_idx:
                Mp = M[:, pi]
                c = {g: neural[g] @ Mp.T / M.shape[1] for g in GROUPS}
                for g in GROUPS:
                    null_max[g].append(c[g].max())
                null_diff.append(c["lateral_parietal"][causal].mean() - c["occipital"][causal].mean())
            entry = {}
            for i, g in enumerate(GROUPS):
                o = obs[g]
                thr = float(np.quantile(null_max[g], 0.95))
                pk = np.unravel_index(np.argmax(o), o.shape)
                entry[g] = {"peak_rho": round(float(o.max()), 4), "peak_eeg_s": round(float(t_eeg[pk[0]]), 3),
                            "peak_model_s": round(float(t_mod[pk[1]]), 3),
                            "p_fwe_peak": float((np.sum(np.array(null_max[g]) >= o.max()) + 1) / (n_perm + 1)),
                            "fwe_threshold": round(thr, 4), "n_sig_cells": int((o > thr).sum()),
                            "causal_mean_rho": round(float(o[causal].mean()), 4)}
                ax = axes[i, j]
                v = max(abs(o).max(), 1e-3)
                ax.imshow(o.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                          extent=[0, t_eeg[-1] + eeg_dt / 2, 0, t_mod[-1] + mdt / 2])
                if (o > thr).any():
                    ax.contour(t_eeg, t_mod, (o > thr).T.astype(float), levels=[0.5], colors="k", linewidths=0.8)
                ax.plot([0, 3], [0, 3], ls=":", c="gray", lw=0.7)
                ax.set_title(f"{lname} | {g.split('_')[0]}\npeak {o.max():.3f}", fontsize=8)
                ax.tick_params(labelsize=7)
                if j == 0:
                    ax.set_ylabel("model time (s)", fontsize=8)
                if i == len(GROUPS) - 1:
                    ax.set_xlabel("EEG time (s)", fontsize=8)
            d_obs = entry["lateral_parietal"]["causal_mean_rho"] - entry["occipital"]["causal_mean_rho"]
            nd = np.array(null_diff)
            entry["lat_minus_occ_causal"] = round(float(d_obs), 4)
            entry["p_lat_minus_occ_two_sided"] = float((np.sum(np.abs(nd - nd.mean()) >= abs(d_obs - nd.mean())) + 1) / (n_perm + 1))
            res["models"][model][lname] = entry
            print(model, lname, json.dumps(entry), flush=True)
        fig.suptitle(f"CT-RSA {model}: Spearman(EEG RDM, model RDM); black = FWE p<0.05", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"ctrsa_{model}.png"), dpi=120)
        plt.close(fig)
    summary(res)
    with open(os.path.join(FIG, "ctrsa.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def summary(res):
    rows = [(m, l, e) for m, ls in res["models"].items() for l, e in ls.items()]
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(rows)), 3.6))
    x = np.arange(len(rows))
    for k, (g, c) in enumerate([("occipital", "C0"), ("lateral_parietal", "C1")]):
        ax.bar(x + (k - 0.5) * 0.4, [e[g]["causal_mean_rho"] for _, _, e in rows], 0.4, color=c, label=g)
    ax.set_xticks(x, [f"{m}\n{l}" for m, l, _ in rows], fontsize=7)
    ax.axhline(0, c="k", lw=0.6)
    ax.set(ylabel="mean Spearman, causal band 0-600 ms", title="CT-RSA by channel group")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ctrsa_summary.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=600)
    ap.add_argument("--perms", type=int, default=500)
    a = ap.parse_args()
    run(a.videos, a.perms)
