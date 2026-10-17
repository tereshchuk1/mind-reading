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


def _tag(names, n_pca, lags, prep, band):
    pc = "-".join(f"{k}{v}" for k, v in n_pca.items()) if isinstance(n_pca, dict) else n_pca
    tag = f"{'+'.join(names)}_pc{pc}_L{'-'.join(map(str, lags))}_lp{prep['lowpass']}_d{prep['dec']}"
    if band == "search":
        tag += "_band-search"
    elif band:
        tag += "_band-" + "-".join(f"{band[n]:.3g}" for n in names)
    return tag


def member_predictions(names, n_pca=30, lags=(0, 30, 2), prep=PREP, cv=True, band=None):
    """Return (test_pred (102,C,T), oof_pred (1000,C,T) or None); cached in scratch/.

    band: None, {feature_set: scale} (banded ridge with fixed scales) or
    "search" (scales searched by banded.search_scales on the training videos;
    inside train_cv the search is repeated within every fold, so the held-out
    videos never influence the scales)."""
    tag = _tag(names, n_pca, lags, prep, band)
    f = path("scratch", f"pred_{tag}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        if "oof" in z or not cv:
            return z["test"], (z["oof"] if "oof" in z else None)
    Y, cnt = group_average(ALL_SUBJECTS, **prep)
    loaded = {n: load_feature(n) for n in names}
    tr = {n: v[0] for n, v in loaded.items()}
    m = cnt > 0

    def fit_predict(fit_idx, feats_tr):
        fs = FeatureSpace(names, n_pca, Y.shape[-1]).fit({n: a[fit_idx] for n, a in feats_tr.items()})
        sc = None
        if band == "search":
            from .banded import search_scales
            Fk = fs.transform(feats_tr)
            sel = fit_idx & m
            sc = search_scales(names, Fk[sel], Y[sel], cnt[sel], fs, lags=range(*lags))
        elif band:
            sc = band
        fs.scales = dict(sc or {})
        Fk = fs.transform(feats_tr)
        sel = fit_idx & m
        return TRF(lags=range(*lags)).fit(Fk[sel], Y[sel], w=cnt[sel]), fs, sc

    trf, fs, sc = fit_predict(np.ones(1000, bool), tr)
    test = trf.predict(fs.transform({n: v[1] for n, v in loaded.items()}))
    out = dict(test=test, scales=np.array(sc, dtype=object))
    print("member", tag, "scales", sc, "alpha", trf.alpha_, flush=True)
    if cv:
        folds = np.random.default_rng(0).permutation(1000) % K
        oof = np.zeros_like(Y)
        for k in range(K):
            # feature space (and band scales) refit inside the fold: no leakage of held-out videos
            trk, fsk, _ = fit_predict(folds != k, tr)
            oof[folds == k] = trk.predict(fsk.transform({n: a[folds == k] for n, a in tr.items()}))
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


def log_result(name, features, params, dev, cv, csv=None):
    """Append one experiment to results/experiments.csv."""
    import csv as _csv
    import datetime

    csv = csv or path("results", "experiments.csv")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    cols = (["name", "features", "params", "dev_top1", "dev_top5"]
            + [f"dev_s{s}_{m}" for s in DEV_SUBJECTS for m in ("top1", "top5")]
            + ["cv_top1", "cv_top5"] + [f"cv_s{s}_{m}" for s in ALL_SUBJECTS for m in ("top1", "top5")] + ["date"])
    row = [name, features, params, round(dev["top1"], 2), round(dev["top5"], 2)]
    row += [v for p in dev["per"] for v in p]
    if cv is None:
        row += [""] * (2 + 2 * len(ALL_SUBJECTS))
    else:
        row += [round(cv["top1"], 2), round(cv["top5"], 2)] + [v for p in cv["per"] for v in p]
    row.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    new = not os.path.exists(csv)
    with open(csv, "a", newline="", encoding="utf-8") as fh:
        wr = _csv.writer(fh)
        if new:
            wr.writerow(cols)
        wr.writerow(row)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        names = arg.split(",")
        te, oof = member_predictions(names)
        print(fmt(dev_test([te], "DEV " + arg)))
        print(fmt(train_cv([oof], "CV  " + arg)), flush=True)
