"""EEG loading and preprocessing.

Key empirical facts that drive the design (see REPORT.md):
* The stimulus-locked signal shared across subjects is dominated by slow
  (< 1 Hz) components, so we do NOT high-pass filter.
* Test recordings have ~0.65x the amplitude of single training trials and
  ~2x higher inter-subject similarity, i.e. they look averaged over repeats.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt

ROOT = os.environ.get("MR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
SFREQ = 500
DEV_SUBJECTS = (1, 2, 3, 4)
HELDOUT_SUBJECTS = (5, 6)
ALL_SUBJECTS = DEV_SUBJECTS + HELDOUT_SUBJECTS
TEST_IDS = np.arange(1001, 1103)


def path(*p):
    return os.path.join(ROOT, *p)


@lru_cache(maxsize=None)
def load_raw(sub: int):
    """Return (train_ids, train_eeg (n,17,1500), test_eeg (102,17,1500)), float64."""
    z = np.load(path("trn", f"sub-{sub:02d}_trn.npz"))
    ids = np.array(sorted(int(k) for k in z.files))
    X = np.stack([z[str(k)] for k in ids]).astype(np.float64)
    T = np.load(path("tst", f"sub-{sub:02d}_tst.npy")).astype(np.float64)
    return ids, X, T


def load_test_labels(sub: int):
    """Ground-truth ids of the test recordings (dev subjects only)."""
    if sub not in DEV_SUBJECTS:
        raise ValueError(f"test labels of sub-{sub:02d} are withheld")
    return np.load(path("tst_stim", f"sub-{sub:02d}_tst_stim.npy")).astype(int)


def preprocess(A: np.ndarray, lowpass: float | None = 20.0, dec: int = 10) -> np.ndarray:
    """Low-pass (no high-pass: the slow drift carries stimulus information) + decimate."""
    if lowpass:
        A = sosfiltfilt(butter(4, lowpass, btype="low", fs=SFREQ, output="sos"), A, axis=-1)
    if dec > 1:
        A = decimate(A, dec, axis=-1, ftype="fir", zero_phase=True)
    return A


def standardize(X: np.ndarray, T: np.ndarray):
    """Per-subject normalisation. Train and test are scaled separately (both
    unsupervised): the mean response over all trials is removed from each split
    (it carries no stimulus identity) and every split gets unit channel variance."""
    def _z(A):
        A = A - A.mean(0, keepdims=True)
        return A / A.std(axis=(0, 2), keepdims=True)
    return _z(X), _z(T)


@lru_cache(maxsize=None)
def load_subject(sub: int, lowpass: float | None = 20.0, dec: int = 10):
    ids, X, T = load_raw(sub)
    X, T = standardize(preprocess(X, lowpass, dec), preprocess(T, lowpass, dec))
    return ids, X.astype(np.float32), T.astype(np.float32)
