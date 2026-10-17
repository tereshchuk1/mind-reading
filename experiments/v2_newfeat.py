"""Step 6: V-JEPA 2 and the new audio encoders inside the TRF, alone and added to
member0's feature sets.

    python experiments/v2_newfeat.py [band]     # band = use banded ridge (scale search)
"""
import os
import sys

from mindreading.data import path
from mindreading.evaluate import dev_test, fmt, log_result, member_predictions, train_cv

M0 = ["lowlevel25", "audio50", "whisper", "alexnet_l10"]
band = "search" if "band" in sys.argv[1:] else None
new = [a for a in sys.argv[1:] if a != "band"] or ["vjepa2", "ast", "hubert", "clap"]
runs = []
for f in new:
    if os.path.exists(path("cache", f"{f}_train.npy")):
        runs += [[f], M0 + [f]]
for sets in runs:
    te, oof = member_predictions(sets, band=band)
    name = "v2_" + ("m0+" + sets[-1] if len(sets) > 1 else sets[0] + "_alone") + ("_banded" if band else "")
    d, c = dev_test([te], name), train_cv([oof], name)
    print(fmt(d)); print(fmt(c), flush=True)
    log_result(name, "+".join(sets), "TRF pca30 lags0-560ms" + (" banded" if band else ""), d, c)
