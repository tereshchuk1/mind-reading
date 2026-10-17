"""Encoding model: stimulus features -> cross-subject EEG response, then
retrieval by correlating the predicted response of every candidate video with
the measured test recording."""
from __future__ import annotations

import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .data import ALL_SUBJECTS, load_subject, path

ALPHAS = np.logspace(0, 7, 29)


# ----------------------------------------------------------------------------- features
def load_feature(name: str):
    tr = np.load(path("cache", f"{name}_train.npy"))
    te = np.load(path("cache", f"{name}_test.npy"))
    return np.nan_to_num(tr), np.nan_to_num(te)


def feature_matrix(names, mode="flat", n_pca: float | int = 0.99, pool: int = 1):
    """Build (1000, D) train and (102, D) test design matrices.

    mode="flat": keep the time axis (each time bin x feature is a regressor), so
                 the ridge can map stimulus dynamics to the EEG time course.
    mode="mean": average over time (static description of the clip).
    Each feature set is z-scored and PCA-reduced on TRAIN videos only, then the
    sets are concatenated with equal total variance.
    """
    blocks_tr, blocks_te = [], []
    for n in names:
        tr, te = load_feature(n)
        if pool > 1 and tr.shape[1] % pool == 0:
            tr = tr.reshape(tr.shape[0], -1, pool, tr.shape[2]).mean(2)
            te = te.reshape(te.shape[0], -1, pool, te.shape[2]).mean(2)
        if mode == "mean":
            tr, te = tr.mean(1), te.mean(1)
        else:
            tr, te = tr.reshape(len(tr), -1), te.reshape(len(te), -1)
        sc = StandardScaler().fit(tr)
        tr, te = sc.transform(tr), sc.transform(te)
        p = PCA(n_components=n_pca, svd_solver="full", random_state=0).fit(tr)
        tr, te = p.transform(tr), p.transform(te)
        s = np.sqrt((tr ** 2).sum(1).mean())
        blocks_tr.append(tr / s)
        blocks_te.append(te / s)
    return np.concatenate(blocks_tr, 1), np.concatenate(blocks_te, 1)


# ----------------------------------------------------------------------------- targets
def group_average(subjects=ALL_SUBJECTS, **prep):
    """Mean training response per video over the given subjects -> (1000,C,T), counts."""
    acc = cnt = None
    for s in subjects:
        ids, X, _ = load_subject(s, **prep)
        if acc is None:
            acc = np.zeros((1000,) + X.shape[1:], np.float64)
            cnt = np.zeros(1000)
        acc[ids - 1] += X
        cnt[ids - 1] += 1
    m = cnt > 0
    acc[m] /= cnt[m, None, None]
    return acc.astype(np.float32), cnt


# ----------------------------------------------------------------------------- ridge
def fit_ridge(F, Y, w=None):
    """Multi-output ridge, alpha by efficient leave-one-out on training videos."""
    r = RidgeCV(alphas=ALPHAS, fit_intercept=True)
    r.fit(F, Y.reshape(len(Y), -1), sample_weight=w)
    return r


def predict(r, F, shape):
    return r.predict(F).reshape((len(F),) + tuple(shape))


# ----------------------------------------------------------------------------- TRF
class FeatureSpace:
    """Per feature set: z-score + PCA over all frames of the TRAIN videos, then
    sample-and-hold up-sampling to the EEG rate. Each set gets unit total
    variance so no set dominates the shared ridge penalty. n_pca: int or
    {feature_set: n_components} (default 30)."""

    def __init__(self, names, n_pca=30, n_out=150):
        self.names, self.n_pca, self.n_out = list(names), n_pca, n_out

    def fit(self, train: dict):
        self.tf = {}
        for n in self.names:
            a = train[n]
            d = a.shape[-1]
            sc = StandardScaler().fit(a.reshape(-1, d))
            k = self.n_pca.get(n, 30) if isinstance(self.n_pca, dict) else self.n_pca
            p = PCA(n_components=min(k, d), random_state=0).fit(sc.transform(a.reshape(-1, d)))
            self.tf[n] = (sc, p)
        return self

    def transform(self, feats: dict) -> np.ndarray:
        out = []
        for n in self.names:
            a = feats[n]
            sc, p = self.tf[n]
            nb, d = a.shape[1:]
            z = p.transform(sc.transform(a.reshape(-1, d))).reshape(len(a), nb, -1)
            hold = np.arange(self.n_out) * nb // self.n_out    # sample-and-hold index
            out.append(z[:, hold] / np.sqrt(p.n_components_))
        return np.concatenate(out, -1).astype(np.float32)


def time_features(names, n_pca=30, n_out=150):
    """(1000,n_out,D) train and (102,n_out,D) test time-resolved design."""
    loaded = {n: load_feature(n) for n in names}
    fs = FeatureSpace(names, n_pca, n_out).fit({n: v[0] for n, v in loaded.items()})
    return fs.transform({n: v[0] for n, v in loaded.items()}), fs.transform({n: v[1] for n, v in loaded.items()})


def lag(F, lags):
    """(n,T,D) -> (n,T,D*len(lags)), zero-padded causal lags (stimulus precedes EEG)."""
    n, T, D = F.shape
    out = np.zeros((n, T, D * len(lags)), F.dtype)
    for i, l in enumerate(lags):
        out[:, l:, i * D:(i + 1) * D] = F[:, : T - l]
    return out


class TRF:
    """Temporal response function: EEG(t) = sum_l W_l F(t-l) + onset-locked bias.
    Shared over all time points, so it generalises from far fewer parameters than
    mapping the whole clip to the whole recording.

    Ridge is solved in closed form from X'X. The penalty is chosen by K-fold CV
    over *videos* (time samples within a clip are autocorrelated, so leave-one-
    sample-out would under-regularise)."""

    def __init__(self, lags=range(0, 30, 2), alphas=ALPHAS, k=5, seed=0):
        self.lags, self.alphas, self.k, self.seed = list(lags), np.asarray(alphas), k, seed

    def _design(self, F):
        X = lag(F, self.lags)
        n, T, _ = X.shape
        onset = np.broadcast_to(np.eye(T, dtype=F.dtype), (n, T, T))  # per-time intercept
        return np.concatenate([X, onset], -1)

    @staticmethod
    def _gram(X, y, w):
        Xw = X * w[:, None, None]
        return np.einsum("ntd,nte->de", Xw, X, optimize=True), np.einsum("ntd,ntc->dc", Xw, y, optimize=True)

    @staticmethod
    def _solve_path(G, b, alphas):
        ev, V = np.linalg.eigh(G)
        Vb = V.T @ b
        return [V @ (Vb / (ev + a)[:, None]) for a in alphas]

    def fit(self, F, Y, w=None):
        n, C, T = Y.shape
        w = np.ones(n) if w is None else np.asarray(w, float)
        X = self._design(F).astype(np.float64)
        y = Y.transpose(0, 2, 1).astype(np.float64)
        folds = np.random.default_rng(self.seed).permutation(n) % self.k
        parts = [self._gram(X[folds == f], y[folds == f], w[folds == f]) for f in range(self.k)]
        G = sum(p[0] for p in parts)
        b = sum(p[1] for p in parts)
        # criterion = the task metric (top-1 + top-5) for identifying held-out
        # videos from their group-average response; MSE favours over-shrinkage.
        err = np.zeros(len(self.alphas))
        for f in range(self.k):
            Ws = self._solve_path(G - parts[f][0], b - parts[f][1], self.alphas)
            Xv, yv = X[folds == f], y[folds == f]
            for i, W in enumerate(Ws):
                S = col_zscore(similarity(yv, Xv @ W))
                sc = rank_scores(S, np.arange(len(S)))
                err[i] -= sc["top1"] + sc["top5"]
        self.cv_err_ = err
        self.alpha_ = float(self.alphas[int(np.argmin(err))])
        self.W = self._solve_path(G, b, [self.alpha_])[0]
        return self

    def predict(self, F):
        return (self._design(F).astype(np.float64) @ self.W).transpose(0, 2, 1).astype(np.float32)


# ----------------------------------------------------------------------------- scoring
def similarity(M, P, win=None, chan_w=None):
    """Pearson correlation between measured (n,C,T) and predicted (m,C,T) responses."""
    if win is not None:
        M, P = M[..., win], P[..., win]
    if chan_w is not None:
        M, P = M * chan_w[:, None], P * chan_w[:, None]
    M = M.reshape(len(M), -1)
    P = P.reshape(len(P), -1)
    M = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    P = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)
    return M @ P.T / M.shape[1]


def col_zscore(S):
    """Unsupervised normalisation of a (recordings x candidates) similarity matrix:
    removes candidate-specific bias (videos whose prediction resembles everything)."""
    return (S - S.mean(0, keepdims=True)) / (S.std(0, keepdims=True) + 1e-9)


def rank_scores(S, truth_cols):
    r = (-S).argsort(1).argsort(1)[np.arange(len(S)), truth_cols] + 1
    return dict(top1=float((r == 1).mean()), top5=float((r <= 5).mean()), mrank=float(np.median(r)))


def ranked_ids(S, cand_ids):
    return np.asarray(cand_ids)[np.argsort(-S, axis=1)].astype(np.int64)
