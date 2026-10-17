# Mind Reading with EEG: audio-visual TRF encoding

Solution for the *Mind Reading Using Machine Learning and EEG* challenge
(National AI Hackathon 2026, SchubKI / Constructor Technology).

Given a 3 s, 17-channel EEG recording and 102 candidate videos, the pipeline
predicts the EEG response each candidate would evoke and ranks the candidates
by how well the predicted response correlates with the measured one.

**Core idea.** A forward *temporal response function* (TRF) maps time-resolved
stimulus features (sound, low-level vision, Whisper base/small and
AlexNet/CLIP embeddings) to the EEG time course with lags of 0-560 ms. One TRF is shared by
all six subjects and trained on the per-video **group-average** response, which
averages out subject-specific noise. An ensemble of five TRFs over different
feature sets is fused after an unsupervised column z-score of the similarity
matrices.

The detailed report (in Russian) is in [REPORT.md](REPORT.md).

## Results

Dev subjects (test recordings of sub-01..04, labels used only for scoring),
produced by exactly the code path that writes the submission:

| Model | top-1 | top-5 |
|---|---|---|
| chance | 1.0% | 4.9% |
| organisers' baseline (AlexNet `features.3`, ridge, sub-01) | 3.9% | 11.8% |
| **this pipeline, mean of sub-01..04** | **15.9%** | **34.6%** |
| per subject: sub-01 / 02 / 03 / 04 | 25.5 / 10.8 / 17.6 / 9.8% | 39.2 / 32.4 / 44.1 / 22.5% |

Full ablations, per-subject numbers, and the train-only cross-validation
(which also covers sub-05/06) are in [REPORT.md](REPORT.md).

### v2 (branch `v2-experiments`, additive; v1 is tagged `v1-submitted`)

v2 adds two ensemble members (member0's features + AST, + V-JEPA 2) and a
**CCA spatial denoiser**: canonical correlation analysis between the ensemble's
out-of-fold TRF predictions and the training EEG; predictions and recordings are
projected onto the first 2 canonical components and back-projected to the 17
channels before scoring. Everything is fit on training recordings only.

| | dev top-1 | dev top-5 | train-CV top-1 | train-CV top-5 |
|---|---|---|---|---|
| v1 (`configs/final.json`) | 15.9% | 34.6% | 5.7% | 18.9% |
| **v2 (`configs/v2.json`)** | **16.2%** | **41.2%** | **6.4%** | **19.9%** |

Every experiment (v1 reference rows included) is logged in
[results/experiments.csv](results/experiments.csv); the v2 section of
[REPORT.md](REPORT.md) discusses them, including the negative results (banded
ridge, whitening CCA/CorrCA, new encoders as single members).

## Reproduce

Requirements: Python 3.11, CPU is enough (feature extraction ~4 h on 12 cores,
mostly Whisper; training + prediction ~15 min).

```bash
conda create -n mindreading python=3.11 -y
conda activate mindreading
pip install -r requirements.txt
```

Put the challenge data in the repository root (not redistributed here):
`stimuli_train/`, `stimuli_test/`, `trn/`, `tst/`, `tst_stim/`.

```bash
cd src
# 1. stimulus features -> cache/ (videos are read directly, no frames stored)
python -m mindreading.features lowlevel25 audio50 alexnet whisper whisper_small clip
# 2. train the ensemble on the training recordings of all 6 subjects -> weights/
python -m mindreading.pipeline train --config ../configs/final.json
# 3. rank candidates, write submission/, score the same code on sub-01..04
python -m mindreading.pipeline predict --config ../configs/final.json
# optional: figures and analyses for the report -> figures/
python -m mindreading.analysis
```

Step 3 alone reproduces the submission from the stored weights in `weights/`
(it still needs the cached features of the test videos from step 1).

### v2

v2 features are extracted on a GPU in a separate environment (modelling stays
in the CPU environment above):

```bash
conda create -n mindreading-gpu python=3.11 -y
conda activate mindreading-gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers opencv-python imageio-ffmpeg scikit-learn joblib scipy
cd src
python -m mindreading.features vjepa2 ast hubert clap whisper_layers alexnet_layers --device cuda
```

Then, in the `mindreading` environment (v1 outputs are never overwritten):

```bash
cd src
python -m mindreading.pipeline all --config ../configs/v2.json --out-weights weights/v2 --out-submission submission/v2
python -m mindreading.ctrsa            # CT-RSA analysis -> figures/ctrsa_*.png, figures/ctrsa.json
```

The training step computes 10-fold out-of-fold predictions of every member to
fit the CCA denoiser (cached in `scratch/`). The experiment scripts behind
`results/experiments.csv` are in `experiments/` (run with `PYTHONPATH=src`).

## Repository layout

| Path | Content |
|---|---|
| `src/mindreading/data.py` | EEG loading, low-pass + decimation, per-subject standardisation |
| `src/mindreading/features.py` | all stimulus feature extractors |
| `src/mindreading/model.py` | feature space (z-score + PCA), TRF ridge with retrieval-based CV, scoring |
| `src/mindreading/evaluate.py` | dev-test and train-CV evaluation of ensemble members |
| `src/mindreading/pipeline.py` | final training / inference / submission writer |
| `src/mindreading/analysis.py` | TRF kernels, time-window analysis, failure cases, noise ceiling |
| `src/mindreading/banded.py` | v2: banded ridge (per-feature-set penalty search) |
| `src/mindreading/cca.py` | v2: CCA / CorrCA spatial filters and denoisers |
| `src/mindreading/ctrsa.py` | v2: cross-temporal RSA by channel group |
| `configs/final.json` | the submitted v1 configuration |
| `configs/v2.json` | v2 configuration |
| `weights/`, `weights/v2/` | trained model weights (feature transforms + TRF weights, joblib); v2 adds `denoise.joblib` |
| `submission/`, `submission/v2/` | `sub-05_ranked_candidate_list.npy`, `sub-06_ranked_candidate_list.npy`, dev scores |
| `results/experiments.csv` | one row per experiment: dev and train-CV top-1/top-5, mean and per subject |
| `experiments/` | v2 experiment scripts |
| `figures/` | figures used in the report |
| `01_preprocess_and_train_model.ipynb`, `02_predict_movies.ipynb` | organisers' baseline notebooks (reference) |
| `03_compare_approaches.ipynb` | earlier exploration harness |

## Pre-trained models (public, released before the hackathon)

| Model | Hugging Face id / source | Revision |
|---|---|---|
| Whisper base (encoder, layers 3 and 6) | `openai/whisper-base` | `e37978b90ca9030d5170a5c07aadb050351a65bb` |
| Whisper small (encoder, layers 4, 8, 12) | `openai/whisper-small` | `973afd24965f72e36ca33b3055d56a652f456b4d` |
| CLIP ViT-B/16 (vision tower, layers 6 and 12, CLS) | `openai/clip-vit-base-patch16` | `57c216476eefef5ab752ec549e440a49ae4ae5f3` |
| AlexNet (`features.3`, `features.10`) | `torchvision.models.alexnet`, `AlexNet_Weights.IMAGENET1K_V1` | torchvision 0.29.1 |
| v2: V-JEPA 2 ViT-L (64 frames, fp16; layers 12 and 24, spatial mean per temporal token) | `facebook/vjepa2-vitl-fpc64-256` | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` |
| v2: AST (AudioSet; layers 6 and 12, frequency mean per time patch) | `MIT/ast-finetuned-audioset-10-10-0.4593` | `f826b80d28226b62986cc218e5cec390b1096902` |
| v2: HuBERT base (layers 6 and 12, 50 Hz) | `facebook/hubert-base-ls960` | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` |
| v2: CLAP HTSAT (audio embedding of 0.5 s windows) | `laion/clap-htsat-unfused` | `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` |
| v2: Whisper base, all encoder hidden states 0-6 (CT-RSA) | `openai/whisper-base` | as above |
| v2: AlexNet, all 5 conv layers (CT-RSA) | torchvision, `IMAGENET1K_V1` | torchvision 0.26.0 (GPU env) |

v2 features were extracted with torch 2.11.0+cu128 and transformers 5.20.1
(`results/hf_revisions.json`).

No pre-trained model is fine-tuned; only the linear TRF is trained.

## Compliance with the rules

* Only **training** recordings (all six subjects) are used to fit anything:
  feature PCA, TRF weights and the ridge penalty (chosen by 5-fold CV over
  training videos with the task metric).
* Test recordings are used only through unsupervised per-subject normalisation
  (mean removal / unit variance) and the column z-score of the similarity
  matrix. No test recording is ever used as a template.
* Dev test labels (sub-01..04) are used only to score and select models.
* No assignment / matching constraint across test recordings (Hungarian,
  Sinkhorn) is used; we tested it and it did not help (see report).
* Rows of the submission follow the original order of `tst/sub-XX_tst.npy`.
* v2's CCA denoiser is fit on out-of-fold predictions and EEG of **training**
  videos only; on test data it is a fixed linear map applied to each recording.
