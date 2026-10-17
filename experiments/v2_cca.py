"""Step 5: CCA (prediction <-> EEG) and CorrCA (between subjects) spatial filters
on top of member0's TRF predictions (and the v1 ensemble mean prediction)."""
import json
import sys

import numpy as np

from mindreading.cca import cca_eval, corrca_eval, topo_table
from mindreading.data import path
from mindreading.evaluate import fmt, log_result, member_predictions

M0 = ["lowlevel25", "audio50", "whisper", "alexnet_l10"]
te, oof = member_predictions(M0)
out = {}
for reg in (0.01, 0.1):
    for k in (4, 8, 17):
        d, c, rho = cca_eval(te, oof, k, reg, tag=f"cca k{k} reg{reg}")
        print(fmt(d)); print(fmt(c), flush=True)
        log_result(f"v2_cca_k{k}_reg{reg}", "+".join(M0), f"CCA(oof pred, group EEG) k={k} shrink={reg}", d, c)
        out["cca_rho_reg%s" % reg] = np.round(rho, 3).tolist()
for reg in (0.01, 0.1):
    for k in (3, 6, 10, 17):
        d, c, ev, A = corrca_eval(te, oof, k, reg, tag=f"corrca k{k} reg{reg}")
        print(fmt(d)); print(fmt(c), flush=True)
        log_result(f"v2_corrca_k{k}_reg{reg}", "+".join(M0), f"CorrCA between subjects, top {k} comps, shrink={reg}", d, c)
        out[f"corrca_eig_reg{reg}"] = np.round(ev, 4).tolist()
        out[f"corrca_topo_reg{reg}"] = topo_table(A, 3)
json.dump(out, open(path("results", "cca.json"), "w"), indent=1)
