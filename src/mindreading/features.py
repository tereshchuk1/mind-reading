"""Stimulus feature extraction.

All features are cached as cache/{name}_train.npy (1000, T, D) and
cache/{name}_test.npy (102, T, D) where T is a time axis over the 3 s clip.

Extractors:
  lowlevel  - luminance / colour / contrast / motion energy / optical flow on a
              4x4 grid, 10 Hz (30 bins). Cheap, CPU only.
  alexnet   - torchvision AlexNet features.3 / features.10, 6 frames, 4x4 pooled.
  clip      - openai/clip-vit-base-patch16 vision tower, 4 Hz (12 frames),
              CLS token of an intermediate and the final layer.
  dinov2    - facebook/dinov2-base, 4 Hz, CLS token of layers 6 and 12.
  audio     - log-mel (32 bands) + broadband envelope + onset strength, 10 Hz.
  whisper   - openai/whisper-base encoder layers 3 and 6, 50 Hz (150 frames).
  whisper_small - openai/whisper-small encoder layers 4, 8 and 12, 50 Hz.
  lowlevel25 / audio50 - the same at 25 Hz / 50 Hz (used by the TRF model).
"""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import numpy as np

from .data import path

CACHE = path("cache")
DUR = 3.0


def video_path(vid: int) -> str:
    split = "stimuli_train" if vid <= 1000 else "stimuli_test"
    return path(split, f"{vid:04d}.mp4")


def read_frames(vid: int, size: int | None = None):
    cap = cv2.VideoCapture(video_path(vid))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if size:
            f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    t = np.arange(len(frames)) / fps
    return np.stack(frames), t


def bin_time(feat: np.ndarray, t: np.ndarray, n_bins: int) -> np.ndarray:
    """Average per-frame features (n_frames, D) into n_bins equal bins over 0-3 s."""
    edges = np.linspace(0, DUR, n_bins + 1)
    out = np.zeros((n_bins, feat.shape[1]), np.float32)
    for b in range(n_bins):
        m = (t >= edges[b]) & (t < edges[b + 1])
        if m.any():
            out[b] = feat[m].mean(0)
        else:  # clip shorter than 3 s: hold last value
            out[b] = feat[min(np.searchsorted(t, edges[b]), len(t) - 1)]
    return out


def _grid(a: np.ndarray, g: int = 4) -> np.ndarray:
    h, w = a.shape[:2]
    return a[: h // g * g, : w // g * g].reshape(g, h // g, g, w // g, *a.shape[2:]).mean((1, 3)).reshape(-1)


def lowlevel_one(vid: int, n_bins: int = 30) -> np.ndarray:
    fr, t = read_frames(vid, size=128)
    fr = fr.astype(np.float32) / 255.0
    lum = fr @ np.array([0.299, 0.587, 0.114], np.float32)
    feats = []
    prev = None
    for i in range(len(fr)):
        L = lum[i]
        g8 = (L * 255).astype(np.uint8)
        if prev is None:
            flow = np.zeros((*L.shape, 2), np.float32)
            diff = np.zeros_like(L)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev, g8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            diff = np.abs(L - lum[i - 1])
        prev = g8
        mag = np.linalg.norm(flow, axis=-1)
        yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]] / L.shape[0] - 0.5
        w = mag + 1e-6
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1)
        edge = np.hypot(gx, gy)
        f = np.concatenate([
            [L.mean(), L.std(), edge.mean(), diff.mean(), mag.mean(),
             flow[..., 0].mean(), flow[..., 1].mean(),
             (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum(),          # motion centroid
             (L * xx).sum() / (L.sum() + 1e-6), (L * yy).sum() / (L.sum() + 1e-6)],         # luminance centroid
            fr[i].reshape(-1, 3).mean(0),                                 # RGB
            _grid(L), _grid(diff), _grid(mag), _grid(edge),               # 4x4 maps
            _grid(flow[..., 0]), _grid(flow[..., 1]),
        ])
        feats.append(f)
    return bin_time(np.asarray(feats, np.float32), t, n_bins)


def audio_one(vid: int, n_bins: int = 30, sr: int = 16000) -> np.ndarray:
    """Log-mel envelope (32 bands) + broadband envelope + onset strength."""
    import torch
    import torchaudio

    y = load_audio(vid, sr)
    hop = sr // 100
    mel = torchaudio.transforms.MelSpectrogram(sr, n_fft=1024, hop_length=hop, n_mels=32)(torch.from_numpy(y.copy()))
    logmel = torch.log(mel + 1e-6).T.numpy()                                  # (frames, 32)
    env = np.log(mel.sum(0).numpy() + 1e-6)[:, None]
    onset = np.maximum(np.diff(env[:, 0], prepend=env[0, 0]), 0)[:, None]
    f = np.concatenate([logmel, env, onset], 1)
    t = np.arange(len(f)) * hop / sr
    return bin_time(f, t, n_bins)


def load_audio(vid: int, sr: int = 16000) -> np.ndarray:
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "quiet", "-i", video_path(vid),
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    y = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32)
    n = int(DUR * sr)
    return np.pad(y[:n], (0, max(0, n - len(y))))


def extract_whisper(name: str = "whisper", model_id: str = "openai/whisper-base", layers=(3, 6), device: str = "cpu"):
    """Whisper encoder hidden states; frames are 20 ms (50 Hz), so the first 150
    frames cover exactly the 3 s clip at the EEG sampling rate used downstream."""
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel

    fe = WhisperFeatureExtractor.from_pretrained(model_id)
    enc = WhisperModel.from_pretrained(model_id).encoder.eval().to(device)
    torch.set_num_threads(os.cpu_count())
    for split, ids in [("train", range(1, 1001)), ("test", range(1001, 1103))]:
        out = os.path.join(CACHE, f"{name}_{split}.npy")
        if os.path.exists(out):
            continue
        res = []
        for vid in ids:
            inp = fe(load_audio(vid), sampling_rate=16000, return_tensors="pt").input_features.to(device)
            with torch.inference_mode():
                hs = enc(inp, output_hidden_states=True).hidden_states
            res.append(np.concatenate([hs[l][0, :150].float().cpu().numpy() for l in layers], -1))
            if vid % 100 == 0:
                print(name, vid, flush=True)
        np.save(out, np.stack(res).astype(np.float32))


def _run_parallel(fn, ids, workers):
    with ProcessPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(fn, ids, chunksize=8)))


def extract_simple(name: str, fn, workers: int = 10):
    os.makedirs(CACHE, exist_ok=True)
    for split, ids in [("train", range(1, 1001)), ("test", range(1001, 1103))]:
        out = os.path.join(CACHE, f"{name}_{split}.npy")
        if os.path.exists(out):
            continue
        arr = _run_parallel(fn, list(ids), workers)
        np.save(out, arr.astype(np.float32))
        print(name, split, arr.shape, flush=True)


def extract_alexnet(n_frames: int = 6):
    """torchvision AlexNet (IMAGENET1K_V1): features.3 (the organisers' baseline
    layer) and features.10 (last conv), each 4x4 average-pooled, n_frames frames
    evenly spaced in 0-3 s -> alexnet_l3 (T,3072) and alexnet_l10 (T,4096)."""
    import torch
    import torch.nn.functional as F
    from torchvision import models

    net = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    times = (np.arange(n_frames) + 0.5) * DUR / n_frames
    for split, ids in [("train", range(1, 1001)), ("test", range(1001, 1103))]:
        outs = {3: [], 10: []}
        if all(os.path.exists(os.path.join(CACHE, f"alexnet_l{k}_{split}.npy")) for k in outs):
            continue
        for vid in ids:
            fr, t = read_frames(vid, size=224)
            idx = [int(np.argmin(np.abs(t - x))) for x in times]
            h = (torch.from_numpy(fr[idx]).permute(0, 3, 1, 2).float() / 255 - mean) / std
            with torch.inference_mode():
                for i, layer in enumerate(net):
                    h = layer(h)
                    if i in outs:
                        outs[i].append(F.adaptive_avg_pool2d(h, 4).flatten(1).numpy())
        for k, v in outs.items():
            np.save(os.path.join(CACHE, f"alexnet_l{k}_{split}.npy"), np.stack(v).astype(np.float32))
        print("alexnet", split, flush=True)


def extract_vit(name: str, model_id: str, layers: tuple[int, ...], n_frames: int = 12, device: str = "cpu"):
    """CLS tokens of the given hidden-state layers for n_frames frames evenly spaced in 0-3 s."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    os.makedirs(CACHE, exist_ok=True)
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)
    if hasattr(model, "vision_model"):  # CLIP: use the vision tower only
        model = model.vision_model
    torch.set_num_threads(os.cpu_count())
    times = (np.arange(n_frames) + 0.5) * DUR / n_frames
    for split, ids in [("train", range(1, 1001)), ("test", range(1001, 1103))]:
        out = os.path.join(CACHE, f"{name}_{split}.npy")
        if os.path.exists(out):
            continue
        res = []
        for vid in ids:
            fr, t = read_frames(vid)
            idx = [int(np.argmin(np.abs(t - x))) for x in times]
            inp = proc(images=list(fr[idx]), return_tensors="pt").to(device)
            with torch.inference_mode():
                hs = model(pixel_values=inp["pixel_values"], output_hidden_states=True).hidden_states
            res.append(np.concatenate([hs[l][:, 0].float().cpu().numpy() for l in layers], -1))
            if vid % 100 == 0:
                print(name, vid, flush=True)
        np.save(out, np.stack(res).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=["lowlevel", "lowlevel25", "audio", "audio50", "alexnet", "whisper", "whisper_small", "clip", "dinov2"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    for w in a.which:
        if w == "lowlevel":
            extract_simple("lowlevel", lowlevel_one)
        elif w == "lowlevel25":
            extract_simple("lowlevel25", partial(lowlevel_one, n_bins=75))
        elif w == "audio":
            extract_simple("audio", audio_one)
        elif w == "audio50":
            extract_simple("audio50", partial(audio_one, n_bins=150))
        elif w == "alexnet":
            extract_alexnet()
        elif w == "whisper":
            extract_whisper(device=a.device)
        elif w == "whisper_small":
            extract_whisper("whisper_small", "openai/whisper-small", layers=(4, 8, 12), device=a.device)
        elif w == "clip":
            extract_vit("clip", "openai/clip-vit-base-patch16", (6, 12), device=a.device)
        elif w == "dinov2":
            extract_vit("dinov2", "facebook/dinov2-base", (6, 12), device=a.device)


if __name__ == "__main__":
    main()
