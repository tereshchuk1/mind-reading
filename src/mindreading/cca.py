"""Spatial filters for retrieval (v2, step 5). Everything is fit on TRAINING data.

(a) CCA between out-of-fold TRF predictions and the measured group-average
    training EEG (samples = time points of all training videos). Candidates are
    then scored by Pearson correlation in the canonical space: recording @ Wy vs
    prediction @ Wx.
(b) CorrCA between subjects on the training videos (components that are
    maximally correlated across subjects watching the same video). Recording and
    prediction are both projected onto the top components before scoring.
    Because the TRF is a multi-output ridge, fitting it on CorrCA-projected EEG
    gives exactly (TRF prediction) @ V for the same alpha, so projecting the
    predictions is equivalent to projecting the EEG before the TRF (up to the
    choice of alpha).

For train_cv the filters are refit inside every fold without the held-out
videos (the OOF predictions of the other folds come from models that did see
the fold, a second-order effect we accept and note in the report).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .data import ALL_SUBJECTS, CHANNELS, DEV_SUBJECTS, load_subject, load_test_labels
from .evaluate import K, PREP, _summ
from .model import col_zscore, group_average, rank_scores, similarity


def _cov(A, B=None):
    """A, B: (n, C, T) -> C x C sum over trials and time of a b'."""
    B = A if B is None else B
    return np.einsum("nct,ndt->cd", A.astype(np.float64), B.astype(np.float64))


def _shrink(R, reg):
    return (1 - reg) * R + reg * np.trace(R) / len(R) * np.eye(len(R))


def cca_filters(P, Y, reg=0.1):
    """Regularised CCA between predictions P and EEG Y (both (n,C,T)).
    Returns Wx (C,C), Wy (C,C), canonical correlations (descending)."""
    P = P - P.mean((0, 2), keepdims=True)
    Y = Y - Y.mean((0, 2), keepdims=True)
    Cxx, Cyy, Cxy = _shrink(_cov(P), reg), _shrink(_cov(Y), reg), _cov(P, Y)
    # generalized eigenproblem for Wy: Cyx Cxx^-1 Cxy w = rho^2 Cyy w
    M = Cxy.T @ np.linalg.solve(Cxx, Cxy)
    ev, Wy = eigh(M, Cyy)
    o = np.argsort(-ev)
    ev, Wy = ev[o], Wy[:, o]
    rho = np.sqrt(np.clip(ev, 0, None))
    Wx = np.linalg.solve(Cxx, Cxy @ Wy) / (rho + 1e-12)
    return Wx, Wy, rho, _cov(P), _cov(Y)


def denoiser(W, R, k):
    """Project onto the first k components and back to channels: x -> x W_k A_k'
    with the forward model A_k = R W_k (W_k' R W_k)^-1 (keeps the channel metric)."""
    Wk = W[:, :k]
    A = R @ Wk @ np.linalg.inv(Wk.T @ R @ Wk)
    return Wk @ A.T


def corrca(ids_list, X_list, reg=0.1, videos=None):
    """CorrCA over subjects. ids_list/X_list: per subject (ids, (n,C,T)).
    videos: optional set of video ids (0-based) to use. Returns V (C,C) sorted by
    ISC eigenvalue, eigenvalues, forward-model topographies A (C,C)."""
    C = X_list[0].shape[1]
    S = np.zeros((1000, C, X_list[0].shape[2]))
    cnt = np.zeros(1000)
    for ids, X in zip(ids_list, X_list):
        keep = np.ones(len(ids), bool) if videos is None else np.isin(ids - 1, videos)
        S[ids[keep] - 1] += X[keep]
        cnt[ids[keep] - 1] += 1
    multi = cnt >= 2           # only videos seen by >= 2 subjects (and in `videos`)
    Rw = np.zeros((C, C))
    for ids, X in zip(ids_list, X_list):
        Rw += _cov(X[multi[ids - 1]])
    Rt = _cov(S[multi])
    Rb = Rt - Rw
    ev, V = eigh(Rb, _shrink(Rw, reg))
    o = np.argsort(-ev)
    ev, V = ev[o], V[:, o]
    A = Rw @ V @ np.linalg.inv(V.T @ Rw @ V)          # forward model (Parra et al.)
    return V, ev, A, Rw


def _list(P):
    """A single prediction array or a list of ensemble members' predictions."""
    return P if isinstance(P, (list, tuple)) else [P]


def _proj(A, W):
    return np.einsum("nct,ck->nkt", A, W)


def dev_test_proj(test_pred, Wx, Wy, tag="", prep=PREP):
    rows = []
    for s in DEV_SUBJECTS:
        _, _, T = load_subject(s, **prep)
        S = sum(col_zscore(similarity(_proj(T, Wy), _proj(P, Wx))) for P in _list(test_pred))
        rows.append(rank_scores(S, load_test_labels(s) - 1001))
    return _summ(rows, tag)


def train_cv_proj(oof, fold_filters, tag="", prep=PREP):
    """fold_filters[k] = (Wx, Wy) fit without fold k."""
    folds = np.random.default_rng(0).permutation(1000) % K
    per = []
    for s in ALL_SUBJECTS:
        ids, X, _ = load_subject(s, **prep)
        acc = {"top1": [], "top5": [], "n": []}
        for k in range(K):
            vids = np.where(folds == k)[0]
            sel = np.isin(ids - 1, vids)
            Wx, Wy = fold_filters[k]
            S = sum(col_zscore(similarity(_proj(X[sel], Wy), _proj(O[vids], Wx))) for O in _list(oof))
            r = rank_scores(S, np.searchsorted(vids, ids[sel] - 1))
            acc["top1"].append(r["top1"]); acc["top5"].append(r["top5"]); acc["n"].append(sel.sum())
        per.append({m: float(np.average(acc[m], weights=acc["n"])) for m in ("top1", "top5")})
    return _summ(per, tag)


def cca_eval(test_pred, oof, n_comp=8, reg=0.1, prep=PREP, tag="cca", backproj=False):
    Y, cnt = group_average(ALL_SUBJECTS, **prep)
    m = cnt > 0
    folds = np.random.default_rng(0).permutation(1000) % K
    oof_mean = np.mean(_list(oof), 0)        # ensemble: filters from the mean prediction

    def filt(mask):
        Wx, Wy, rho, Rx, Ry = cca_filters(oof_mean[mask], Y[mask], reg)
        if backproj:
            return denoiser(Wx, Rx, n_comp), denoiser(Wy, Ry, n_comp), rho
        return Wx[:, :n_comp], Wy[:, :n_comp], rho

    Wx, Wy, rho = filt(m)
    d = dev_test_proj(test_pred, Wx, Wy, tag, prep)
    ff = [filt(m & (folds != k))[:2] for k in range(K)]
    c = train_cv_proj(oof, ff, tag, prep)
    return d, c, rho


def corrca_eval(test_pred, oof, n_comp=8, reg=0.1, prep=PREP, tag="corrca", backproj=False):
    subs = [load_subject(s, **prep) for s in ALL_SUBJECTS]
    ids_list, X_list = [s[0] for s in subs], [s[1] for s in subs]
    folds = np.random.default_rng(0).permutation(1000) % K

    def filt(videos):
        V, ev, A, Rw = corrca(ids_list, X_list, reg, videos=videos)
        W = denoiser(V, Rw, n_comp) if backproj else V[:, :n_comp]
        return W, ev, A

    W, ev, A = filt(None)
    d = dev_test_proj(test_pred, W, W, tag, prep)
    ff = []
    for k in range(K):
        Wk = filt(np.where(folds != k)[0])[0]
        ff.append((Wk, Wk))
    c = train_cv_proj(oof, ff, tag, prep)
    return d, c, ev, A


def topo_table(A, n=3):
    """Named-channel forward-model topographies of the first n components."""
    out = {}
    for i in range(n):
        a = A[:, i] / np.abs(A[:, i]).max()
        a = a * np.sign(a[np.argmax(np.abs(a))])
        out[f"C{i + 1}"] = {c: round(float(v), 2) for c, v in zip(CHANNELS, a)}
    return out
