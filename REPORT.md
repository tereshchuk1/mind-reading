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

## 10. Final pipeline

This section summarises the submitted pipeline end to end and shows what each step contributes. All fitted components use training recordings only. Test recordings enter only through unsupervised normalisation.

### 10.1 Overview

```
video + audio ──► feature extraction ──► 7 TRF encoders ──► predicted EEG for each candidate
                                                                      │
test EEG ──► preprocessing ──► CorrCA denoiser ◄──────────────────────┘
                                     │
                                     ▼
                Pearson similarity ──► column z-score ──► ensemble mean ──► ranking
```

### 10.2 Steps

1. **EEG preprocessing.** 20 Hz low-pass with no high-pass (the slow components carry stimulus information), decimation to 50 Hz (150 samples per 3 s recording). For each subject and split, the mean response is removed and every channel is scaled to unit variance.
2. **Training target.** For each of the 1000 training videos, the average of the preprocessed responses of all subjects who saw it. Averaging suppresses subject-specific noise and matches the higher SNR of the averaged test recordings.
3. **Stimulus features.** Time-resolved descriptions of each clip:
   - **sound:** log-mel spectrum, loudness envelope and onsets; Whisper base and small; AST;
   - **vision:** luminance, contrast, motion and optical flow; AlexNet; CLIP; V-JEPA 2.

   Each set is reduced by PCA to 30 components (fitted on training videos) and resampled to 50 Hz. Pre-trained networks are frozen.
4. **TRF encoders.** Seven linear temporal response functions map the features, with lags of 0–560 ms, to the 17-channel EEG. All share the core set (low-level video + sound + Whisper + AlexNet) and differ in what is added: CLIP, Whisper small, AST, V-JEPA 2, or a smaller Whisper PCA. The ridge penalty is chosen by 5-fold CV over training videos, using the task metric as the criterion.
5. **CorrCA denoiser.** From the training EEG of the six subjects, CorrCA finds the spatial components most correlated across people watching the same video. Two are kept: lateral parietal (P8, P7, PO8, PO7) and occipital (Oz, O1, O2). Recordings and predictions are projected onto them and back to the 17 channels, which removes activity that is not reproducible between people.
6. **Scoring and ranking.** For every test recording and every candidate, we compute the Pearson correlation between the recording and the predicted response (17 channels × 150 samples). Each candidate's column is z-scored across the 102 recordings, the seven matrices are averaged, and the candidates are sorted. Rows keep the original order of the test file.

### 10.3 Contribution of each step

| Pipeline | dev top-1 / top-5 (%) | train-CV top-1 / top-5 (%) |
|---|---|---|
| organisers' baseline (sub-01) | 3.9 / 11.8 | – |
| one TRF: low-level video + sound + AlexNet | 12.7 / 29.2 | 4.0 / 15.6 |
| + Whisper | 15.0 / 32.1 | 5.5 / 17.8 |
| ensemble of 5 TRFs | 15.9 / 34.6 | 5.7 / 18.9 |
| ensemble of 7 TRFs (+ AST, + V-JEPA 2) | 16.4 / 35.0 | 6.1 / 19.4 |
| **+ CorrCA denoiser (final)** | **17.4 / 40.9** | **5.9 / 20.3** |

Final model per development subject (top-1 / top-5, %): sub-01 22.5 / 48.0, sub-02 15.7 / 38.2, sub-03 16.7 / 47.1, sub-04 14.7 / 30.4. On train-CV, the held-out subjects score sub-05 6.0 / 20.0 and sub-06 3.0 / 17.0, in the same range as the development subjects. The final model reaches about 80% (top-1) and 85% (top-5) of the noise ceiling, i.e. the score obtained using other subjects' test responses as templates (21.8 / 48.2).

The largest gains come from the temporal model (TRF instead of a whole-clip regression), from sound features (Whisper), and from spatial denoising, which mainly improves top-5.

### 10.4 Model selection

Each step was kept only if it improved both validation schemes: the dev test set (408 averaged recordings) and train-CV (about 5,600 single-trial rankings from all six subjects, never touching test data). For the last step, the CorrCA denoiser was chosen over a CCA-based denoiser (16.2 / 41.2 on dev). The two are equivalent on train-CV, and CorrCA has the higher dev top-1. Because dev labels took part in the selection, the final dev score is slightly optimistic.

### 10.5 What did not help

- **Per-subject models** (5.9 / 15.0): one person's data is too little; the shared response matters more.
- **Backward model**, reconstructing features from EEG (≈1.2 / 7.8).
- **Hungarian or Sinkhorn assignment** (no gain; also a grey zone of the rules).
- **Banded ridge** with a separate penalty per feature set (14.5 / 31.9).
- **Scoring in CCA/CorrCA component space** without projecting back to channels (≤ 11.3 / 29.2).
- **New encoders added inside a single TRF** instead of as separate ensemble members (top-1 dropped to 11.8–13.7).
- **HuBERT and CLAP** as extra members (no gain on train-CV).

