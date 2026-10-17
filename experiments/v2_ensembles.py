"""Step 6b: v1 ensemble + new-encoder members, with and without CCA denoising (k=2)."""
import json

from mindreading.cca import cca_eval
from mindreading.data import path
from mindreading.evaluate import dev_test, fmt, log_result, member_predictions, train_cv

cfg = json.load(open(path("configs", "final.json")))
v1 = [member_predictions(m["features"], m.get("n_pca", 30), tuple(m.get("lags", [0, 30, 2])), cfg["prep"])
      for m in cfg["members"]]
M0 = cfg["members"][0]["features"]
extra = {f: member_predictions(M0 + [f]) for f in ("ast", "vjepa2", "hubert", "clap")}
combos = {"v1+ast": ["ast"], "v1+vjepa2": ["vjepa2"], "v1+ast+vjepa2": ["ast", "vjepa2"],
          "v1+ast+vjepa2+hubert+clap": ["ast", "vjepa2", "hubert", "clap"]}
for name, add in combos.items():
    mem = v1 + [extra[a] for a in add]
    tes, oofs = [m[0] for m in mem], [m[1] for m in mem]
    d, c = dev_test(tes, name), train_cv(oofs, name)
    print(fmt(d)); print(fmt(c), flush=True)
    log_result("v2_ens_" + name, name, "mean col-z Pearson", d, c)
    d, c, _ = cca_eval(tes, oofs, 2, 0.1, tag=name + " cca2", backproj=True)
    print(fmt(d)); print(fmt(c), flush=True)
    log_result("v2_ens_" + name + "_cca2", name, "CCA back-projection k=2 on every member", d, c)
