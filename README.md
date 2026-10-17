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

## Repository layout

| Path | Content |
|---|---|
| `src/mindreading/data.py` | EEG loading, low-pass + decimation, per-subject standardisation |
| `src/mindreading/features.py` | all stimulus feature extractors |
| `src/mindreading/model.py` | feature space (z-score + PCA), TRF ridge with retrieval-based CV, scoring |
| `src/mindreading/evaluate.py` | dev-test and train-CV evaluation of ensemble members |
| `src/mindreading/pipeline.py` | final training / inference / submission writer |
| `src/mindreading/analysis.py` | TRF kernels, time-window analysis, failure cases, noise ceiling |
| `configs/final.json` | the submitted configuration |
| `weights/` | trained model weights (feature transforms + TRF weights, joblib) |
| `submission/` | `sub-05_ranked_candidate_list.npy`, `sub-06_ranked_candidate_list.npy`, dev scores |
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
