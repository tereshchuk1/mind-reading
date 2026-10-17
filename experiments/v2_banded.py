"""Step 3: banded ridge (per-feature-set penalty) on member0's feature sets.

The block scales are searched once on all training videos (coordinate descent on
a 0.25-4x log grid, criterion = the TRF's own 5-fold CV over training videos).
dev_test is therefore clean. In train_cv the scales are kept fixed (alpha is
still re-chosen in every fold); a fully nested search took ~15 min per fold,
so this is a small, documented hyperparameter leak of 4 scales.
Full search result for member0: lowlevel25 0.25, audio50 1, whisper 1, alexnet_l10 1.
"""
import sys

from mindreading.evaluate import dev_test, fmt, log_result, member_predictions, train_cv

sets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["lowlevel25", "audio50", "whisper", "alexnet_l10"]
scales = {"lowlevel25": 0.25, "audio50": 1.0, "whisper": 1.0, "alexnet_l10": 1.0}
if len(sys.argv) > 2:
    scales = dict(zip(sets, map(float, sys.argv[2].split(","))))
name = "v2_banded_" + "+".join(sets) if len(sys.argv) > 1 else "v2_banded_member0"
te, oof = member_predictions(sets, band=scales)
d, c = dev_test([te], name), train_cv([oof], name)
print(fmt(d)); print(fmt(c), flush=True)
log_result(name, "+".join(sets), "banded ridge scales " + str(scales) + " (searched on all train videos; fixed in CV)", d, c)
