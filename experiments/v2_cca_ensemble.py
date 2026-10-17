"""Step 5c: CCA / CorrCA denoising applied to every member of the v1 ensemble
(and to member0 with k=3)."""
import json

from mindreading.cca import cca_eval, corrca_eval
from mindreading.data import path
from mindreading.evaluate import fmt, log_result, member_predictions

cfg = json.load(open(path("configs", "final.json")))
tes, oofs = zip(*[member_predictions(m["features"], m.get("n_pca", 30), tuple(m.get("lags", [0, 30, 2])), cfg["prep"])
                  for m in cfg["members"]])
tes, oofs = list(tes), list(oofs)
for k in (2, 3, 4):
    d, c, _ = cca_eval(tes, oofs, k, 0.1, tag=f"ens cca-denoise k{k}", backproj=True)
    print(fmt(d)); print(fmt(c), flush=True)
    log_result(f"v2_ens_cca_denoise_k{k}", "configs/final.json (5 members)", f"CCA back-projection k={k} on every member", d, c)
    d, c, _, _ = corrca_eval(tes, oofs, k, 0.1, tag=f"ens corrca-denoise k{k}", backproj=True)
    print(fmt(d)); print(fmt(c), flush=True)
    log_result(f"v2_ens_corrca_denoise_k{k}", "configs/final.json (5 members)", f"CorrCA back-projection k={k} on every member", d, c)
M0 = cfg["members"][0]["features"]
for f, name in ((cca_eval, "cca"), (corrca_eval, "corrca")):
    r = f(tes[0], oofs[0], 3, 0.1, tag=f"m0 {name}-denoise k3", backproj=True)
    print(fmt(r[0])); print(fmt(r[1]), flush=True)
    log_result(f"v2_{name}_denoise_k3", "+".join(M0), f"{name} back-projection to channels, k=3 shrink=0.1", r[0], r[1])
