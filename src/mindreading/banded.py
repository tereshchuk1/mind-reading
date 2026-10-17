"""Banded ridge for the TRF: every feature set gets its own penalty.

Scaling feature block j by s_j before a shared ridge with penalty alpha is
equivalent to a ridge with penalty alpha / s_j^2 on that block. The scales are
searched by coordinate descent on a small log grid; the criterion is the same
5-fold CV over training videos (top-1 + top-5 of identifying held-out videos
from their group-average response) that TRF.fit uses to pick alpha. Fold Gram
matrices are computed once; a scale vector c only rescales them (c G c, c b).
"""
from __future__ import annotations

import numpy as np

from .model import TRF, col_zscore, rank_scores, similarity

GRID = (0.25, 0.5, 1.0, 2.0, 4.0)


class ScaleSearch:
    def __init__(self, F, Y, w, block_sizes, lags=range(0, 30, 2), k=5, seed=0):
        trf = TRF(lags=lags, k=k, seed=seed)
        self.trf, self.sizes = trf, list(block_sizes)
        n = len(Y)
        w = np.ones(n) if w is None else np.asarray(w, float)
        self.X = trf._design(F).astype(np.float64)
        self.y = Y.transpose(0, 2, 1).astype(np.float64)
        self.folds = np.random.default_rng(seed).permutation(n) % k
        self.parts = [trf._gram(self.X[self.folds == f], self.y[self.folds == f], w[self.folds == f]) for f in range(k)]
        self.G = sum(p[0] for p in self.parts)
        self.b = sum(p[1] for p in self.parts)
        self.nl, self.D = len(trf.lags), sum(block_sizes)
        self.n_onset = self.X.shape[-1] - self.nl * self.D

    def colscale(self, s):
        feat = np.concatenate([np.full(k, v) for k, v in zip(self.sizes, s)])
        return np.concatenate([np.tile(feat, self.nl), np.ones(self.n_onset)])

    def score(self, s):
        """CV top-1 + top-5 (summed over folds, best alpha) for block scales s."""
        c = self.colscale(s)
        tot = np.zeros(len(self.trf.alphas))
        for f in range(self.trf.k):
            G = (self.G - self.parts[f][0]) * c[:, None] * c[None]
            b = (self.b - self.parts[f][1]) * c[:, None]
            Xv, yv = self.X[self.folds == f] * c, self.y[self.folds == f]
            for i, W in enumerate(self.trf._solve_path(G, b, self.trf.alphas)):
                sc = rank_scores(col_zscore(similarity(yv, Xv @ W)), np.arange(len(yv)))
                tot[i] += sc["top1"] + sc["top5"]
        return float(tot.max())

    def search(self, rounds=2, grid=GRID, verbose=False):
        s = np.ones(len(self.sizes))
        best = self.score(s)
        hist = [(s.tolist(), best)]
        for _ in range(rounds):
            changed = False
            for j in range(len(s)):
                bs = s
                for g in grid:
                    if g == 1.0:
                        continue
                    t = s.copy()
                    t[j] = s[j] * g
                    v = self.score(t)
                    hist.append((t.tolist(), v))
                    if v > best + 1e-9:
                        best, bs, changed = v, t, True
                s = bs
                if verbose:
                    print("  block", j, "scales", np.round(s, 3).tolist(), "cv", round(best / self.trf.k, 4), flush=True)
            if not changed:
                break
        self.best_, self.history_ = best / self.trf.k, hist
        return s


def search_scales(names, F, Y, w, fs, lags=range(0, 30, 2), **kw):
    """Return {feature_set: scale} for a fitted FeatureSpace fs."""
    sizes = [fs.tf[n][1].n_components_ for n in names]
    s = ScaleSearch(F, Y, w, sizes, lags=lags).search(**kw)
    return dict(zip(names, map(float, s)))
