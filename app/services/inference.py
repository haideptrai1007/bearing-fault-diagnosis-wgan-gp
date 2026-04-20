"""
Inference service — non-streaming, single POST returns everything at once.
"""
import base64
import io
import logging
import os
import random
import time
import traceback
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from src.data.cwru_data import cwru_inference
from src.features.cwt import cwt2scalogram
from src.features.stft import stft2spectrogram
from src.models.onnx_inference import ONNXModel

logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────
BASE = r"D:\Capstone\models\ONNX\CNN"

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "MobileNetV4": {
        "scalogram":   rf"{BASE}\SCALOGRAM\MobileNetV4\best_model_int8.onnx",
        "spectrogram": rf"{BASE}\SPECTROGRAM\MobileNetV4\best_model_int8.onnx",
    },
    "TinyNetD": {
        "scalogram":   rf"{BASE}\SCALOGRAM\TinyNetD\best_model_int8.onnx",
        "spectrogram": rf"{BASE}\SPECTROGRAM\TinyNetD\best_model_int8.onnx",
    },
    "EdgeNeXtXXS": {
        "scalogram":   rf"{BASE}\SCALOGRAM\EdgenextXXS\best_model_int8.onnx",
        "spectrogram": rf"{BASE}\SPECTROGRAM\EdgenextXXS\best_model_int8.onnx",
    },
    "GhostNetV3": {
        "scalogram":   rf"{BASE}\SCALOGRAM\GhostNetV3\best_model_int8.onnx",
        "spectrogram": rf"{BASE}\SPECTROGRAM\GhostNetV3\best_model_int8.onnx",
    },
}

TRANSFORM_FN = {
    "scalogram":   cwt2scalogram,
    "spectrogram": stft2spectrogram,
}

# Must match the window size used in cwru_inference / your training pipeline
WINDOW_SIZE = 2048

# ── Label encoding ────────────────────────────────────────────────────────────
LABEL_CLASSES = ["Normal", "OR", "IR", "B"]
_LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABEL_CLASSES)}

LABEL_DISPLAY = {
    0: "Normal Condition",
    1: "Outer Race Fault",
    2: "Inner Race Fault",
    3: "Ball Fault",
}

def _encode_gt(raw_gt) -> int:
    if isinstance(raw_gt, (int, np.integer)):
        return int(raw_gt)
    s = str(raw_gt).strip()
    if s in _LABEL_TO_IDX:
        return _LABEL_TO_IDX[s]
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"Unknown label '{s}'. Add it to LABEL_CLASSES.")

# ── Plot helpers ──────────────────────────────────────────────────────────────
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

def plot_raw_signal(signal: np.ndarray, fs: int = 12000) -> str:
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, signal, linewidth=0.6, color="#0A84FF")
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_title("Raw Vibration Signal", fontsize=10, fontweight="600")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.tick_params(labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)

def plot_spectrogram_from_signal(signal: np.ndarray, fs: int = 12000) -> str:
    """signal should already be the exact window to plot (WINDOW_SIZE samples)."""
    from scipy.signal import stft
    chunk = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    f, t, Zxx = stft(chunk, fs=fs, window="hann", nperseg=256, noverlap=128)
    spec = np.log1p(np.abs(Zxx))
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 3))
    pcm = ax.pcolormesh(t, f / 1000, spec, shading="auto", cmap="inferno")
    fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    ax.set_ylabel("Freq (kHz)", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title("Spectrogram (STFT)", fontsize=10, fontweight="600")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)

def plot_scalogram_from_signal(signal: np.ndarray, fs: int = 12000) -> str:
    """signal should already be the exact window to plot (WINDOW_SIZE samples)."""
    import pywt
    chunk = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    scales = np.arange(1, 129)
    coeffs, freqs = pywt.cwt(chunk, scales, "morl", sampling_period=1/fs)
    scalo = np.log1p(np.abs(coeffs))
    scalo = (scalo - scalo.min()) / (scalo.max() - scalo.min() + 1e-8)
    times = np.arange(len(chunk)) / fs
    fig, ax = plt.subplots(figsize=(10, 3))
    pcm = ax.pcolormesh(times, freqs / 1000, scalo, shading="auto", cmap="plasma")
    fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    ax.set_ylabel("Freq (kHz)", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title("Scalogram (CWT)", fontsize=10, fontweight="600")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_base64(fig)

# ── Data loading ──────────────────────────────────────────────────────────────
def _load_raw_signal(signal_path: str) -> np.ndarray:
    from scipy.io import loadmat
    mat = loadmat(signal_path)
    for k in mat:
        if k.startswith("__"): continue
        if "DE_time" in k or "FE_time" in k or "BA_time" in k:
            return np.asarray(mat[k]).squeeze()
    for k, v in mat.items():
        if k.startswith("__"): continue
        arr = np.asarray(v).squeeze()
        if arr.ndim == 1 and arr.size > 100:
            return arr
    raise ValueError(f"No 1D vibration signal found in {signal_path}")

def _load_signal_images(signal_path: str, transform_key: str) -> Tuple[np.ndarray, np.ndarray]:
    transform_fn = TRANSFORM_FN[transform_key]
    normal = "Normal" in signal_path
    imgs, gts = cwru_inference(signal_path, transform_fn, normal=normal)
    return imgs, gts

# ── Model cache ───────────────────────────────────────────────────────────────
_MODEL_CACHE: Dict[str, ONNXModel] = {}

def _get_model(path: str) -> ONNXModel:
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = ONNXModel(path)
    return _MODEL_CACHE[path]

def _infer_one(arch: str, transform_key: str, x: np.ndarray, gt: int) -> Dict:
    model_path = MODEL_REGISTRY[arch][transform_key]
    try:
        if not os.path.exists(model_path):
            return {"model_name": arch, "transform": transform_key, "label": -1,
                    "ground_truth": gt, "confidence": 0.0, "inference_time": 0.0,
                    "correct": False, "error": f"Not found: {model_path}"}
        model = _get_model(model_path)
        t0 = time.perf_counter()
        logits = model.predict(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        prob = softmax(logits, axis=-1).squeeze()
        label = int(np.argmax(prob))
        confidence = float(np.max(prob))
        return {"model_name": arch, "transform": transform_key, "label": label,
                "ground_truth": gt, "confidence": confidence,
                "inference_time": elapsed_ms, "correct": label == gt, "error": None}
    except Exception as e:
        logger.error("[infer] %s/%s failed: %s", arch, transform_key, traceback.format_exc())
        return {"model_name": arch, "transform": transform_key, "label": -1,
                "ground_truth": gt, "confidence": 0.0, "inference_time": 0.0,
                "correct": False, "error": str(e)}

# ── Main entry point ──────────────────────────────────────────────────────────
def run_all(signal_path: str) -> Dict:
    """
    Load signal, generate 3 plots, run all 8 models, return everything at once.
    """
    logger.info("[run_all] signal_path=%s", signal_path)

    raw = _load_raw_signal(signal_path)
    logger.info("[run_all] raw signal: %d samples", len(raw))

    scal_imgs, scal_gts = _load_signal_images(signal_path, "scalogram")
    spec_imgs, spec_gts = _load_signal_images(signal_path, "spectrogram")
    logger.info("[run_all] scal=%s spec=%s", scal_imgs.shape, spec_imgs.shape)

    n   = min(len(scal_imgs), len(spec_imgs))
    idx = random.randint(0, n - 1)
    gt  = _encode_gt(scal_gts[idx])
    logger.info("[run_all] idx=%d gt=%d", idx, gt)

    # Extract the exact window that will be inferred — all 3 plots show this window
    start  = idx * WINDOW_SIZE
    end    = start + WINDOW_SIZE
    window = raw[start:end] if end <= len(raw) else raw[-WINDOW_SIZE:]
    logger.info("[run_all] plotting window samples [%d:%d]", start, min(end, len(raw)))

    t_plot_start     = time.perf_counter()
    raw_plot         = plot_raw_signal(window)
    spectrogram_plot = plot_spectrogram_from_signal(window)
    scalogram_plot   = plot_scalogram_from_signal(window)
    plot_time_ms     = (time.perf_counter() - t_plot_start) * 1000.0
    logger.info("[run_all] plot time: %.1f ms", plot_time_ms)

    scal_imgs = (scal_imgs / 255.0).astype(np.float32)
    spec_imgs = (spec_imgs / 255.0).astype(np.float32)
    x_scal = scal_imgs[idx:idx + 1]
    x_spec = spec_imgs[idx:idx + 1]

    results: List[Dict] = []
    for arch in MODEL_REGISTRY:
        results.append(_infer_one(arch, "scalogram",   x_scal, gt))
        results.append(_infer_one(arch, "spectrogram", x_spec, gt))
        logger.info("[run_all] %s done", arch)

    return {
        "segment_idx":     idx,
        "ground_truth":    gt,
        "plot_time_ms":    round(plot_time_ms, 1),
        "raw_plot":        raw_plot,
        "spectrogram_plot": spectrogram_plot,
        "scalogram_plot":  scalogram_plot,
        "results":         results,
    }