"""v1 reference rows for results/experiments.csv: member0 and the frozen 5-member ensemble."""
import json

from mindreading.data import path
from mindreading.evaluate import dev_test, fmt, log_result, member_predictions, train_cv

cfg = json.load(open(path("configs", "final.json")))
tes, oofs = [], []
for i, mem in enumerate(cfg["members"]):
    te, oof = member_predictions(mem["features"], mem.get("n_pca", 30), tuple(mem.get("lags", [0, 30, 2])), cfg["prep"])
    tes.append(te); oofs.append(oof)
    if i == 0:
        d, c = dev_test([te], "v1 member0"), train_cv([oof], "v1 member0")
        print(fmt(d)); print(fmt(c), flush=True)
        log_result("v1_member0", "+".join(mem["features"]), "pca30 lags0-560ms ridge", d, c)
d, c = dev_test(tes, "v1 ensemble"), train_cv(oofs, "v1 ensemble")
print(fmt(d)); print(fmt(c))
log_result("v1_ensemble", "configs/final.json (5 members)", "mean col-z Pearson", d, c)
