"""Two complementary validation schemes for an ensemble member (a TRF on a list
of feature sets):

dev_test : fit on all training videos, score the test recordings of the four
           dev subjects (labels used only for scoring). 408 recordings that are
           averaged over repeats, like the held-out test set.
train_cv : 10-fold CV over TRAIN videos. The member is refit without the fold,
           and every single-trial recording of all six subjects is ranked among
           the ~100 held-out videos (~5,600 rankings, includes sub-05/06, never
           touches test data). Out-of-fold predictions are cached so ensembles
           can be scored without refitting.

    python -m mindreading.evaluate lowlevel25,audio50,alexnet_l10 audio50flux
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .data import ALL_SUBJECTS, DEV_SUBJECTS, load_subject, load_test_labels, path
from .model import TRF, FeatureSpace, col_zscore, group_average, load_feature, rank_scores, similarity

PREP = dict(lowpass=20.0, dec=10)
K = 10


def _summ(rows, tag):
    t1 = np.mean([r["top1"] for r in rows]) * 100
    t5 = np.mean([r["top5"] for r in rows]) * 100
    return dict(tag=tag, top1=t1, top5=t5, per=[(round(r["top1"] * 100, 1), round(r["top5"] * 100, 1)) for r in rows])


def fmt(d):
    return f"{d['tag']:52s} top1 {d['top1']:5.1f} top5 {d['top5']:5.1f} | " + " ".join(f"{a:.0f}/{b:.0f}" for a, b in d["per"])


def member_predictions(names, n_pca=30, lags=(0, 30, 2), prep=PREP, cv=True):
    """Return (test_pred (102,C,T), oof_pred (1000,C,T) or None); cached in scratch/."""
    tag = f"{'+'.join(names)}_pc{n_pca}_L{'-'.join(map(str, lags))}_lp{prep['lowpass']}_d{prep['dec']}"
    f = path("scratch", f"pred_{tag}.npz")
    if os.path.exists(f):
        z = np.load(f)
        if "oof" in z or not cv:
            return z["test"], (z["oof"] if "oof" in z else None)
    Y, cnt = group_average(ALL_SUBJECTS, **prep)
    loaded = {n: load_feature(n) for n in names}
    tr = {n: v[0] for n, v in loaded.items()}
    fs = FeatureSpace(names, n_pca, Y.shape[-1]).fit(tr)
    F, Fte = fs.transform(tr), fs.transform({n: v[1] for n, v in loaded.items()})
    m = cnt > 0
    test = TRF(lags=range(*lags)).fit(F[m], Y[m], w=cnt[m]).predict(Fte)
    out = dict(test=test)
    if cv:
        folds = np.random.default_rng(0).permutation(1000) % K
        oof = np.zeros_like(Y)
        for k in range(K):
            trn = (folds != k) & m
            # feature space refit inside the fold as well (no leakage of held-out videos)
            fsk = FeatureSpace(names, n_pca, Y.shape[-1]).fit({n: a[folds != k] for n, a in tr.items()})
            Fk = fsk.transform(tr)
            oof[folds == k] = TRF(lags=range(*lags)).fit(Fk[trn], Y[trn], w=cnt[trn]).predict(Fk[folds == k])
        out["oof"] = oof
    os.makedirs(path("scratch"), exist_ok=True)
    np.savez(f, **out)
    return out["test"], out.get("oof")


def dev_test(preds, tag="", weights=None, prep=PREP):
    weights = weights or [1.0] * len(preds)
    rows = []
    for s in DEV_SUBJECTS:
        _, _, T = load_subject(s, **prep)
        S = sum(w * col_zscore(similarity(T, P)) for w, P in zip(weights, preds))
        rows.append(rank_scores(S, load_test_labels(s) - 1001))
    return _summ(rows, tag)


def train_cv(oofs, tag="", weights=None, prep=PREP):
    weights = weights or [1.0] * len(oofs)
    folds = np.random.default_rng(0).permutation(1000) % K
    per = []
    for s in ALL_SUBJECTS:
        ids, X, _ = load_subject(s, **prep)
        acc = {"top1": [], "top5": [], "n": []}
        for k in range(K):
            vids = np.where(folds == k)[0]
            sel = np.isin(ids - 1, vids)
            S = sum(w * col_zscore(similarity(X[sel], O[vids])) for w, O in zip(weights, oofs))
            r = rank_scores(S, np.searchsorted(vids, ids[sel] - 1))
            acc["top1"].append(r["top1"]); acc["top5"].append(r["top5"]); acc["n"].append(sel.sum())
        per.append({m: float(np.average(acc[m], weights=acc["n"])) for m in ("top1", "top5")})
    return _summ(per, tag)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        names = arg.split(",")
        te, oof = member_predictions(names)
        print(fmt(dev_test([te], "DEV " + arg)))
        print(fmt(train_cv([oof], "CV  " + arg)), flush=True)
