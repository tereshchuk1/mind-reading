# Report: EEG "mind reading" with an audio-visual temporal response function (TRF)

## 1. Summary

We predict **which EEG response each of the 102 candidate videos would evoke**
and rank the videos by the correlation between the predicted and the recorded
response. The prediction is made by a linear *temporal response function*
(TRF): time-varying sound and image features are convolved with kernels of
0–560 ms and produce a 17-channel EEG signal. One TRF is shared by all six
subjects and is trained on the **subject-averaged** response to every training
video. The final model is an ensemble of 7 such TRFs on different feature sets;
before scoring, recordings and predictions pass through a **CorrCA denoiser**
that keeps the 2 spatial components most reproducible *across people*
(section 10).

| Model (dev: test recordings of sub-01..04, 408 recordings) | top-1 | top-5 |
|---|---|---|
| chance | 1.0% | 4.9% |
| organisers' baseline (AlexNet `features.3` + ridge, sub-01) | 3.9% | 11.8% |
| ridge "whole clip → whole recording" (no TRF), sound + video | 5.4% | 22.3% |
| TRF, low-level video + sound + AlexNet | 12.7% | 29.2% |
| TRF + Whisper (one ensemble member) | 15.0% | 32.1% |
| v1: ensemble of 5 TRFs | 15.9% | 34.6% |
| v2: + AST and V-JEPA 2 members, CCA denoiser (section 10) | 16.2% | 41.2% |
| **final: 7 TRFs + CorrCA denoiser k=2 (submitted)** | **17.4%** | **40.9%** |

Final model per subject: sub-01 22.5% / 48.0%, sub-02 15.7% / 38.2%,
sub-03 16.7% / 47.1%, sub-04 14.7% / 30.4% (v1: 25.5 / 39.2, 10.8 / 32.4,
17.6 / 44.1, 9.8 / 22.5).

The predictions for sub-05 and sub-06 are in `submission/`. Estimate for the
held-out subjects from cross-validation on their training data (single trials,
~100 candidates), final model: sub-05 6% / 20%, sub-06 3% / 17%, versus
5–10% / 19–25% for the dev subjects. sub-05/06 therefore behave like the dev
subjects, and a result of the same order as on dev should be expected.

## 2. Data: what we use and why

**What data exploration showed (this shaped the whole architecture):**

1. **The response to a video is very similar across people.** The correlation
   between single trials of two subjects for the same video is ≈0.07–0.11,
   for different videos ≈0. The stimulus-driven part of the EEG is therefore
   mostly *shared* and can be learned from all six subjects at once.
2. **The test recordings appear to be averaged over repetitions.** Test
   amplitude is ≈0.65 of a single training trial, and the inter-subject
   similarity for the same video is twice as high (≈0.17–0.23). The training
   target is therefore an average as well: the mean response of the six
   subjects to each training video (higher SNR, closer to the test data).
3. **The information is slow and spread over the whole 3 s.** The 0.1–0.5 Hz band is
   present in the data and carries a slow stimulus-driven signal, and our
   additional 0.5 Hz high-pass simply cut it out. We therefore **apply no
   high-pass**, only a 20 Hz low-pass and decimation to 50 Hz. The window is the
   whole 0–3 s, not 50–600 ms as in the baseline.

## 3. EEG preprocessing

1. 4th-order Butterworth low-pass at 20 Hz (zero-phase), decimation
   500 → 50 Hz (FIR anti-aliasing) → 150 samples per recording.
2. Separately for every subject and every split: subtraction of the mean
   response over all trials and normalisation of every channel to unit variance. This
   compensates for the different amplitude of train (single trials) and test
   (averaged) and for differences between subjects.
3. Training target: mean of the normalised responses of all subjects who saw
   the video (weights = number of subjects).

Alternatives tested (TRF, dev top-1/top-5): low-pass 10 Hz — 12.0/28.9;
no filter — 11.8/28.9; 40 Hz at 100 Hz — 12.0/28.7; **20 Hz — 12.7/29.2**.
The differences are within noise; 20 Hz was chosen as the ERP standard.

## 4. Feature engineering (stimulus features)

All features are **time series** over 0–3 s, not one number per clip: the brain
responds to *events* (sound onsets, cuts, motion), and their timing matters.

| Set | What it is | Rate | Dimension |
|---|---|---|---|
| `audio50` | log-mel spectrum (32 bands) + broadband loudness envelope + onset strength (positive derivative of the envelope) | 50 Hz | 34 |
| `whisper` | `openai/whisper-base` encoder, layers 3 and 6 (20 ms frames) | 50 Hz | 1024 |
| `whisper_small` | `openai/whisper-small` encoder, layers 4, 8, 12 | 50 Hz | 2304 |
| `ast` | AST (AudioSet), layers 6 and 12, frequency mean per time patch | ≈10 Hz | 1536 |
| `lowlevel25` | luminance, contrast, edges, frame difference, optical flow (Farneback), centres of mass of motion and luminance, RGB, 4×4 maps of luminance/motion/flow/edges | 25 Hz | 110 |
| `alexnet_l10` | AlexNet (ImageNet) `features.10`, 4×4 pooling, 6 frames | 2 Hz | 4096 |
| `clip` | CLIP ViT-B/16, CLS token of layers 6 and 12, 12 frames | 4 Hz | 1536 |
| `vjepa2` | V-JEPA 2 ViT-L (64 frames), layers 12 and 24, spatial mean per temporal token | ≈10 Hz | 2048 |

Every set is z-scored and reduced by PCA to 30 components (fitted on the
frames of the training videos only); all sets are then scaled to the same total
variance (so that none dominates the shared ridge penalty) and upsampled to
50 Hz by sample-and-hold.

## 5. Model: why a TRF and how it is trained

**TRF:** EEG(t, channel) = Σ_lag W[lag] · F(t − lag) + b(t), lags 0, 40, …,
560 ms, plus a separate intercept for every time point (the response to the
clip onset itself). The kernels W are shared across time points, so there are
few parameters (≈1500 × 17), and the model generalises to new videos much better
than ridge "whole clip → whole recording" (5.4% → 12.7% top-1 with the same
features).

**Training (fine-tuning):** the pre-trained networks (Whisper, AST, AlexNet,
CLIP, V-JEPA 2) are **not fine-tuned**; they are frozen feature extractors.
With 1000 videos, fine-tuning a network on EEG overfits. Only the TRF ridge regression
is trained, in closed form. α is chosen by 5-fold CV **over training videos**
(not over samples: neighbouring samples within a clip are correlated, and
leave-one-out under-regularises), and the selection criterion is **the task
metric** (top-1 + top-5 when identifying held-out videos).

**Ranking:** Pearson correlation between the recorded response and the
predicted response of every candidate over all 17 channels × 150 samples; then
every column (candidate) is z-scored over the 102 recordings, which removes
"universal" candidates that resemble everything (12.7% versus 10.3% without
this normalisation). Ensemble: mean of the z-scored similarity matrices of the
7 models. In the final model recordings and predictions are first passed
through the CorrCA denoiser (section 10).

## 6. Why these models: biological rationale

**Ablations (TRF; dev top-1/top-5; train-CV on single trials top-1/top-5):**

| Features | dev | train-CV |
|---|---|---|
| low-level video only | 4.2 / 12.5 | – |
| AlexNet only | 2.9 / 13.2 | – |
| CLIP only | 3.9 / 11.5 | – |
| all visual (low-level + AlexNet) | 4.4 / 16.9 | 2.6 / 11.2 |
| sound envelope/spectrum only | 8.3 / 22.1 | 2.8 / 10.8 |
| Whisper only | 10.5 / 27.2 | 3.8 / 13.7 |
| sound + video | 12.7 / 29.2 | 4.0 / 15.6 |
| sound + Whisper + video | 15.0 / 32.1 | 5.5 / 17.8 |

1. **Sound features dominate the phase-locked EEG, with a purely posterior
   montage.** The organisers gave the channel order: Pz P3 P7 O1 Oz O2 P4 P8
   P1 P5 PO7 PO3 POz PO4 PO8 P6 P2, i.e. only parietal, parieto-occipital and
   occipital electrodes, **no frontal or central sites**. The classic vertex
   auditory N1–P2 (maximal at Cz/FCz) is barely visible here, so the earlier
   explanation "auditory cortex → vertex" is wrong. Still, sound alone gives
   twice as much as all visual features. Two (not mutually exclusive)
   explanations are possible:
   * **Auditory response at lateral parietal sites.** Auditory-cortex dipoles
     (superior temporal gyrus) also project onto the lateral posterior
     electrodes P7/P8/PO7/PO8 (closest to temporal cortex; the N1/P2
     polarity inversion relative to the vertex is seen there). This agrees with
     the topography of the TRF kernels: the energy of the sound kernels is
     highest at **P8, P7, PO8, PO7** (audio50: P8 1.00, PO8 0.88, P7 0.87;
     Whisper: P7 1.00, PO7 0.91), while for low-level video it is highest at
     **Oz, O2, PO8** (`figures/trf_kernels.png`). The differences are moderate
     (0.5–1.0 across all channels), i.e. volume conduction is large.
   * **Alternative hypothesis: sound partly acts as a proxy for visual
     events.** In video clips sound onsets coincide with cuts, impacts and
     objects appearing, i.e. with abrupt image changes to which visual cortex
     (occipital channels) responds. Part of the "auditory" information would
     then really be a visual response that the sound features describe more
     precisely than our visual features (6 AlexNet frames, 25 Hz low-level).
     CT-RSA by channel group helps separate the hypotheses (section 10.6).
   The TRF kernels for sound features peak at **120–200 ms**, the latency of
   N1/P2 (and of the visual N1).
2. **Whisper as a model of the "auditory hierarchy".** The envelope describes
   only primary auditory cortex. Intermediate Whisper layers encode phonetics
   and sound categories, i.e. what non-primary auditory cortex (superior
   temporal gyrus) processes. Whisper gives the largest gain over the simple
   features (12.7% → 15.0% top-1); speech and non-speech clips were not
   compared separately.
3. **Vision adds independent information** (sound 8.3% → sound + video
   12.7%). Low-level features (luminance, contrast, motion) and the AlexNet
   convolutional layers help roughly equally: visual responses in the EEG
   (P1/N1, ~100–200 ms, occipital channels) are driven mainly by changes of
   luminance and contrast. Semantic embeddings (CLIP) add almost nothing:
   high-level semantics is hard to see in single-trial scalp EEG with
   17 channels.
4. **Multisensory integration.** Sound and video add up almost linearly, as
   expected for a linear sum of auditory- and visual-cortex responses at the
   scalp; non-linear interaction (superior temporal sulcus) is not captured by
   the model.
5. **One model for all people.** The inter-subject similarity of the responses
   and the failure of per-subject TRFs (5.9% versus 12.7% for the shared one)
   show that the stimulus-driven response is mostly shared (as in
   inter-subject correlation studies with films), and differences between
   people are noise. Averaging over subjects is the cheapest way to suppress
   that noise. The final CorrCA denoiser uses the same idea in space: it keeps
   only the spatial components that are reproducible between people.
6. **Dynamics over time** (`figures/time_windows.png`): identification is best
   from the first 0.5 s (top-5 25%, the response to the clip onset), but every
   following 0.5 s window still gives 13–19% top-5. The brain keeps tracking
   events throughout the clip, so the 0–3 s window beats the 50–600 ms window.

## 7. What did not work (negative results)

| Idea | Result (dev top-1/top-5) | Conclusion |
|---|---|---|
| ridge "whole clip → whole recording" | 5.4 / 22.3 | too many parameters; the TRF is 2× better |
| per-subject TRF / mix with the shared one | 5.9 / 15.0; 8.8 / 25.0 | one person's data is too little; the shared response matters more |
| backward model (reconstructing features from EEG) | ≈1.2 / 7.8, worse in the ensemble | unlike in auditory-attention decoding, it loses to the forward model here |
| Hungarian / Sinkhorn (each video once) | 11.5 / 29.7; 13.2 / 30.1 | does not help; also a grey zone of the rules |
| CSLS | 11.3 / 30.9 | on par with column z-scoring |
| longer lags up to 960 ms | 11.5 / 28.9 | worse; 0–560 ms is enough |
| Whisper-small instead of base | 12.3 / 33.6 | not better than base; kept as an ensemble member |
| channel weighting by reliability, projection onto the prediction subspace | 5.6 / 21.3; ≤5.4 / 18.1 | no gain |
| neural TRF with a contrastive loss | not finished on CPU | earlier contrastive models reached 2.2 / 9.1 |

## 8. Error analysis and limitations

* **Errors depend on the video, not on the subject**: the ranks of the correct
  video correlate across dev subjects (ρ = 0.26). Clips with **steady,
  monotonous sound** are identified worst (ρ of rank with loudness variance
  −0.28, with onset strength −0.15): without sound events there is no evoked
  response. Loudness itself is only weakly related to errors (ρ = +0.19: loud
  steady sounds are slightly worse); quiet clips are *not* worse than others
  (that hypothesis was not confirmed), and the amount of motion in the frame
  has almost no effect (ρ ≈ 0). When the model predicts a weak response (small
  amplitude), the video is usually identified poorly (ρ = −0.29).
* **Large spread between subjects**: in v1, sub-01 ≈25% top-1 and sub-04 ≈9%
  (final: 22.5% and 14.7%). A similar spread is likely for sub-05/06.
* **Ceiling.** Even an "ideal" template (the test responses of other subjects)
  reaches only ≈22% / 48%; the rest is EEG noise. The final model reaches ≈80%
  of the ceiling for top-1 and ≈85% for top-5.
* The TRF kernel has increased energy at the last lag (560 ms), a sign that the
  model describes part of the slow (<1 Hz) components as a "constant shift";
  longer lags did not help.
* Only 17 posterior channels: there are no fronto-central sites where auditory
  responses are maximal, so the auditory component is seen only through the
  lateral P7/P8/PO7/PO8 and volume conduction.
* The linear model does not capture non-linear multisensory integration.
  V-JEPA 2 features were extracted on a GPU (without fine-tuning); large video
  models were not fine-tuned.

## 9. Reproducibility

All steps are described in `README.md`: feature extraction → `pipeline train`
→ `pipeline predict`. The weights of the final model (`weights/member0-6.joblib`
and `weights/denoise.joblib`, ≈8 MB) are in the repository; public models are
listed with their exact Hugging Face revisions. `pipeline predict` writes the
submission and at the same time runs the identical code on sub-01..04 with known
labels (a check of the format and of the row order). A full retrain from
scratch reproduces the submitted files byte for byte.

## 10. v2 and the final model: additional experiments

v2 was built **additively**: v1 is frozen under the tag `v1-submitted` (commit
`13215f6`); its files are now in `weights/v1/`, `submission/v1/`,
`configs/v1.json`. v2 is in `weights/v2/`, `submission/v2/`, `configs/v2.json`.
The final model (10.0) is in `weights/`, `submission/`, `configs/final.json`.
Every experiment is a row in `results/experiments.csv` (dev and train-CV, mean
and per subject, including sub-05/06). During v2 a change was accepted only if
it improved **both** dev **and** train-CV (standard error of dev top-1 ≈1.7 pp).

### 10.0 Final choice: v2's 7 members + CorrCA instead of CCA

Only one variant can be submitted, and the goal was the highest test metric.
Test labels exist only for the dev subjects, so the final model was chosen as
the best dev result among all variants (on the same 7 members as v2):

| v2's 7 members + … | dev top-1 / top-5 | train-CV top-1 / top-5 | train-CV sub-05 | train-CV sub-06 |
|---|---|---|---|---|
| no denoiser | 16.4 / 35.0 | 6.1 / 19.4 | – | – |
| CCA k=2 (v2) | 16.2 / 41.2 | 6.4 / 19.9 | 6.9 / 19.3 | 3.9 / 16.1 |
| CorrCA k=3 | 14.0 / 36.5 | – | – | – |
| **CorrCA k=2 (final)** | **17.4 / 40.9** | 5.9 / 20.3 | 6.0 / 20.0 | 3.0 / 17.0 |

CorrCA k=2 gives the highest dev top-1 (+1.2 pp over v2) with a top-5 0.3 pp
lower; on train-CV the two variants are equal (sum of top-1 + top-5: 26.2 vs
26.3). This is within noise, and the choice is partly fitted to dev, a
deliberate trade-off for the highest metric. Advantages of CorrCA: the filter
is built from the training EEG of the six subjects only (no out-of-fold
predictions needed); biologically it directly implements the idea of a
response shared between people (inter-subject correlation); and its
components are interpretable (section 10.2: lateral parietal and occipital).

### 10.1 v1 versus v2

v2 = v1's 5 members + member0's features with **AST** + member0's features
with **V-JEPA 2**, and a **CCA denoiser**: CCA between the ensemble's
out-of-fold predictions and the group-average training EEG; predictions and
recordings are projected onto the first 2 canonical components (r = 0.43,
0.34) and back-projected to the 17 channels.

| dev (top-1 / top-5, %) | mean | sub-01 | sub-02 | sub-03 | sub-04 |
|---|---|---|---|---|---|
| v1 ensemble | 15.9 / 34.6 | 25.5 / 39.2 | 10.8 / 32.4 | 17.6 / 44.1 | 9.8 / 22.5 |
| v2 | 16.2 / 41.2 | 21.6 / 47.1 | 13.7 / 36.3 | 15.7 / 44.1 | 13.7 / 37.3 |
| **final** | **17.4 / 40.9** | 22.5 / 48.0 | 15.7 / 38.2 | 16.7 / 47.1 | 14.7 / 30.4 |

| train-CV (single trials, %) | mean | sub-01 | sub-02 | sub-03 | sub-04 | sub-05 | sub-06 |
|---|---|---|---|---|---|---|---|
| v1 ensemble | 5.7 / 18.9 | 8.0 / 23.7 | 4.9 / 17.6 | 6.0 / 20.7 | 5.1 / 17.5 | 5.3 / 17.5 | 4.9 / 16.4 |
| v2 | 6.4 / 19.9 | 10.3 / 25.6 | 5.3 / 20.7 | 6.1 / 19.4 | 5.8 / 18.6 | 6.9 / 19.3 | 3.9 / 16.1 |
| final | 5.9 / 20.3 | 10 / 25 | 5 / 20 | 6 / 21 | 6 / 19 | 6 / 20 | 3 / 17 |

v2's gain over v1 is mainly in **top-5** (+6.6 pp on dev, ≈2.7 standard
errors; +1.0 on train-CV); top-1 on dev is within noise (+0.3), +0.7 on
train-CV. The gain is uneven: sub-02/04 and sub-05 improve, sub-01's top-1
drops, sub-06's train-CV is slightly worse. Decomposition of v2's gain (rows
`v2_ens_*` in the CSV):

| Variant | dev | train-CV |
|---|---|---|
| v1 ensemble | 15.9 / 34.6 | 5.7 / 18.9 |
| + CCA denoiser k=2 | 15.9 / 40.2 | 5.8 / 19.3 |
| + AST and V-JEPA 2 members, no denoiser | 16.4 / 35.0 | 6.1 / 19.4 |
| + both (v2) | 16.2 / 41.2 | 6.4 / 19.9 |
| + also HuBERT and CLAP (not used: 2 more members, no gain on CV) | 16.7 / 41.2 | 6.4 / 19.9 |

Caveat: the configuration was chosen from ≈10 ensemble variants with dev
involved in the choice, which is why agreement with train-CV (which never
touches test data) was required for v2. k=2 components is also the optimum on
train-CV (k=3, 4 are worse). The v2 submission agrees with v1 on the top-1
choice for only ≈1/3 of the sub-05/06 recordings, so it is a noticeably
different model, not a cosmetic change.

### 10.2 Channels and biology

The organisers gave the montage (posterior electrodes only) and preprocessing
(0.1–40 Hz, blink ICA, 1000 → 500 Hz). This changed the interpretation
(sections 2 and 6): auditory responses are visible only at lateral
P7/P8/PO7/PO8, and sound may partly act as a proxy for visual events. Channel
names are now in `data.py` (`CHANNELS`) and on `figures/trf_kernels.png`.

**CorrCA between subjects** (training data) finds three components with the
highest inter-subject correlation, with easily named topographies (forward
model): C1 (λ = 0.55): **lateral parietal P8, P7, PO8, PO7, P6**; C2 (0.38):
**occipital Oz, O1, O2**; C3 (0.22): **midline parietal Pz, P1, P2, P4**
(`results/cca.json`). The final denoiser keeps C1 and C2. The signal most
reproducible across people is lateral parietal, which is compatible with an
auditory (temporal) source but does not prove it.

### 10.3 CCA and CorrCA as spatial filters

| Variant (member0) | dev | train-CV |
|---|---|---|
| no filter | 15.0 / 32.1 | 5.5 / 17.8 |
| CCA, scoring in canonical space, k = 4 / 8 / 17 | ≤ 9.3 / 26.0 | ≤ 4.4 / 14.0 |
| CorrCA, projection onto top-k components (k = 3…17) | ≤ 11.3 / 29.2 | ≤ 5.1 / 16.0 |
| CCA denoiser (projection and back to channels), k = 2 | 14.2 / 35.3 | 5.5 / 18.7 |
| CorrCA denoiser, k = 2 | 14.2 / 37.7 | 5.3 / 18.4 |
| v1 ensemble + CCA denoiser, k = 2 | 15.9 / 40.2 | 5.8 / 19.3 |
| v1 ensemble + CorrCA denoiser, k = 2 | 16.4 / 39.7 | 5.6 / 19.4 |

Scoring directly in component space is **worse** even with k = 17: such a
projection whitens the data and amplifies noisy low-variance directions,
whereas Pearson correlation over the original channels implicitly weights the
strong shared patterns. Denoising (top-k components, back to the channels via
the forward model) keeps the channel metric and removes non-reproducible
directions, which is what raised top-5. Projecting the EEG onto components
*before* the TRF is equivalent to projecting the predictions *after* it (ridge
is linear in its outputs), up to the choice of α, so the TRFs were not refit.
In train-CV the filters are refit in every fold without the held-out videos
(the out-of-fold predictions of the other folds come from models that saw this
fold, a second-order effect).

### 10.4 New encoders (GPU environment `mindreading-gpu`)

| Features | dev, alone | train-CV, alone | dev, + member0 | train-CV, + member0 |
|---|---|---|---|---|
| (member0 without additions) | – | – | 15.0 / 32.1 | 5.5 / 17.8 |
| V-JEPA 2 ViT-L (64 frames) | 2.2 / 12.3 | 2.8 / 11.9 | 11.8 / 33.1 | 5.4 / 18.6 |
| AST (AudioSet) | 9.8 / 29.4 | 3.6 / 14.1 | 13.2 / 37.0 | 5.1 / 17.6 |
| HuBERT base | 9.1 / 20.8 | 3.2 / 11.4 | 12.5 / 32.8 | 5.0 / 17.4 |
| CLAP (0.5 s windows) | 4.9 / 15.9 | 2.6 / 10.7 | 13.7 / 33.6 | 4.9 / 17.6 |
| for comparison: Whisper base alone (v1) | 10.5 / 27.2 | 3.8 / 13.7 | | |

* AST alone is almost as good as Whisper (AudioSet sound events rather than
  speech seem to matter more for the EEG); the speech model HuBERT is worse;
  CLAP with 0.5 s windows is too coarse in time.
* V-JEPA 2 alone is at the AlexNet level: 32 temporal tokens (~94 ms) averaged
  over space do not give the TRF the information it needs. It fit in 6 GB of
  VRAM with 64 frames (1.7 GB, fp16), so the number of frames did not have to
  be reduced.
* Inside a single TRF any addition **lowers top-1** (more features → stronger
  regularisation), but as **separate ensemble members** AST and V-JEPA 2 add
  diversity and improve both validation schemes.

### 10.5 Banded ridge (negative result)

A separate ridge penalty for every feature set (equivalent to scaling the block
before the shared ridge), searched by coordinate descent over a 0.25–4× grid
with 5-fold CV on training videos: the search found "weaken lowlevel25 by 4×",
everything else unchanged. Result: dev 14.5 / 31.9, train-CV 5.4 / 17.2, worse
than member0 (15.0 / 32.1, 5.5 / 17.8). A fully nested search (repeated in every
fold) took ≈15 min per fold with 16 GB of RAM, so the scales are fixed in
train-CV (a small hyper-parameter leak that would work *in favour* of the
method, and there is still no gain). Conclusion: the equal variance of the
feature sets in v1 is already close to optimal, and the selection criterion
(top-1 + top-5 on 200 videos) is too noisy to tune 4 penalties.

### 10.6 CT-RSA: which model layers are reflected in the EEG, and when

Training data only (does not affect the submission): 600 videos, neural RDMs
on 60 ms windows from two group averages (subjects 1–3 versus 4–6,
cross-validated correlation distance), model RDMs per layer and clip time,
Spearman "EEG time × model time", 500 video permutations (FWE via the matrix
maximum). Channel groups: occipital (O1 Oz O2 PO3 POz PO4) and lateral parietal
(P7 P8 PO7 PO8 P5 P6). Figures `figures/ctrsa_*.png`, numbers
`figures/ctrsa.json`.

| Model / layer | peak ρ, occipital | peak ρ, lateral | mean ρ (0–600 ms), lateral − occipital | p of the difference |
|---|---|---|---|---|
| Whisper L0 (input) | 0.064 | 0.058 | −0.003 | 0.006 |
| Whisper L3 | 0.045 | 0.045 | −0.003 | 0.014 |
| Whisper L6 (output) | 0.037 | 0.037 | −0.002 | 0.030 |
| AlexNet conv1 | 0.023 | 0.010 | −0.001 | 0.50 |
| AlexNet conv2 | 0.067 | 0.033 | −0.001 | 0.55 |
| AlexNet conv5 | 0.052 | 0.044 | −0.005 | 0.002 |
| V-JEPA 2 middle / last | 0.029 / 0.028 | 0.030 / 0.033 | −0.004 / −0.003 | 0.002 / 0.028 |

* All peaks are significant (FWE p < 0.01) and fall in the **early response to
  the clip onset: EEG 90–210 ms**. Along model time the structure is almost
  vertical: clip content changes little over 3 s, so the model RDMs are
  similar at any moment, and the diagonal (tracking of events within the clip)
  is weak.
* For Whisper the early layers (spectrum) match the EEG better than the deep
  ones; for AlexNet the middle layers (conv2–conv4) are stronger than conv1.
  The noise ceiling (split-half, Spearman–Brown) is highest in the 120–180 ms
  window (0.14 occipital, 0.12 lateral) and small on average (≈0.015).
* **The hypothesis "Whisper → lateral, AlexNet → occipital" was only partly
  confirmed.** AlexNet is clearly "occipital" (the lateral peak is 1.2–2.3×
  lower, conv1 is almost not significant laterally). Whisper is equally strong
  in both groups (peak ratio ≈0.9–1.0), and in the causal window even slightly
  stronger occipitally. Sound information is topographically less "occipital"
  than visual information (compatible with a lateral/temporal contribution and
  with CorrCA C1), but it is clearly visible in occipital channels too, which
  supports the alternative hypothesis that the sound features partly describe
  visual events.

### 10.7 What did not work in v2

| Idea | Result (dev; train-CV) | Conclusion |
|---|---|---|
| banded ridge | 14.5 / 31.9; 5.4 / 17.2 | worse than equal set weights |
| CCA/CorrCA projection without going back to channels | ≤ 11.3 / 29.2 | whitening amplifies noise |
| V-JEPA 2, AST, HuBERT, CLAP inside member0's TRF | top-1 drops to 11.8–13.7 | useful only as separate ensemble members |
| HuBERT and CLAP as ensemble members | no gain on train-CV | not included |
| denoising member0 alone | top-5 rises, top-1 drops | works together with the ensemble |
