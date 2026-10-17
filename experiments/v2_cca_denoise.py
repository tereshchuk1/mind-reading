"""Step 5b: CCA / CorrCA as denoisers (top-k components back-projected to channels)."""
from mindreading.cca import cca_eval, corrca_eval
from mindreading.evaluate import fmt, log_result, member_predictions

M0 = ["lowlevel25", "audio50", "whisper", "alexnet_l10"]
te, oof = member_predictions(M0)
# denoising variant: project onto the top-k components and back to channels
for k in (2, 4, 8, 12):
    d, c, _ = cca_eval(te, oof, k, 0.1, tag=f"cca-denoise k{k}", backproj=True)
    print(fmt(d)); print(fmt(c), flush=True)
    log_result(f"v2_cca_denoise_k{k}", "+".join(M0), f"CCA back-projection to channels, k={k} shrink=0.1", d, c)
    d, c, _, _ = corrca_eval(te, oof, k, 0.1, tag=f"corrca-denoise k{k}", backproj=True)
    print(fmt(d)); print(fmt(c), flush=True)
    log_result(f"v2_corrca_denoise_k{k}", "+".join(M0), f"CorrCA back-projection to channels, k={k} shrink=0.1", d, c)
