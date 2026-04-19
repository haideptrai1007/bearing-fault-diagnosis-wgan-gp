"""
Inference service for vibration fault diagnosis.

Handles:
- Plot generation (raw signal, spectrogram, scalogram) as base64 PNGs
- Single-model inference
- Multi-model inference (8 models) for the /predict-all endpoint
- A streaming generator that yields plots first, then each model result
"""
import logging
import traceback
import base64
import io
import os
import random
import time
from typing import Dict, Generator, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend (no GUI)

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from src.data.cwru_data import cwru_inference
from src.features.cwt import cwt2scalogram
from src.features.stft import stft2spectrogram
from src.models.onnx_inference import ONNXModel


# --------------------------------------------------------------------------- #
# Model registry — 4 architectures × 2 transforms = 8 ONNX files.
# Use absolute paths directly; do NOT wrap with os.path.join() because
# os.path.join(base, absolute_path) discards `base` on Windows and behaves
# inconsistently on Linux with Windows-style paths.
# --------------------------------------------------------------------------- #
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "MobileNetV4": {
        "scalogram":   r"D:\Capstone\models\ONNX\CNN\SCALOGRAM\MobileNetV4\best_model.onnx",
        "spectrogram": r"D:\Capstone\models\ONNX\CNN\SPECTROGRAM\MobileNetV4\best_model.onnx",
    },
    "MobileOneS0": {
        "scalogram":   r"D:\Capstone\models\ONNX\CNN\SCALOGRAM\MobileOneS0\best_model.onnx",
        "spectrogram": r"D:\Capstone\models\ONNX\CNN\SPECTROGRAM\MobileOneS0\best_model.onnx",
    },
    "EdgeNeXtXXS": {
        "scalogram":   r"D:\Capstone\models\ONNX\CNN\SCALOGRAM\EdgenextXXS\best_model.onnx",
        "spectrogram": r"D:\Capstone\models\ONNX\CNN\SPECTROGRAM\EdgenextXXS\best_model.onnx",
    },
    "GhostNetV3": {
        "scalogram":   r"D:\Capstone\models\ONNX\CNN\SCALOGRAM\GhostNetV3\best_model.onnx",
        "spectrogram": r"D:\Capstone\models\ONNX\CNN\SPECTROGRAM\GhostNetV3\best_model.onnx",
    },
}

# cwt2scalogram / stft2spectrogram are passed INTO cwru_inference as a
# transform callback — they operate on raw segments, not on image arrays.
# They are NOT used here for plotting.  Plotting has its own dedicated
# implementations below (plot_spectrogram_from_signal, plot_scalogram_from_signal).
TRANSFORM_FN = {
    "scalogram":   cwt2scalogram,    # used only inside _load_signal_images
    "spectrogram": stft2spectrogram, # used only inside _load_signal_images
}

# --------------------------------------------------------------------------- #
# Label encoding                                                               #
# cwru_inference returns string labels (e.g. 'Normal', 'Ball-07') OR          #
# integer labels depending on implementation. We normalise to int here.        #
# The order defines the class index — must match your training label encoding. #
# --------------------------------------------------------------------------- #
# Labels exactly as returned by cwru_inference
LABEL_CLASSES = ["Normal", "OR", "IR", "B"]
_LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABEL_CLASSES)}


def _encode_gt(raw_gt) -> int:
    """
    Convert a ground-truth value from cwru_inference to an integer class index.

    Handles:
      - Already an int/np.integer  → return as-is
      - A string like 'Normal'     → look up in LABEL_CLASSES
      - A numpy string (np.str_)   → same as string
    """
    if isinstance(raw_gt, (int, np.integer)):
        return int(raw_gt)
    s = str(raw_gt).strip()
    if s in _LABEL_TO_IDX:
        return _LABEL_TO_IDX[s]
    # Fallback: try parsing as integer string e.g. '3'
    try:
        return int(s)
    except ValueError:
        raise ValueError(
            f"Unknown label '{s}'. Add it to LABEL_CLASSES in inference.py."
        )


# --------------------------------------------------------------------------- #
# Plot helpers                                                                 #
# --------------------------------------------------------------------------- #
def _fig_to_base64(fig) -> str:
    """Convert a Matplotlib figure to a base64 PNG data URL."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def plot_raw_signal(signal: np.ndarray, fs: int = 12000) -> str:
    """Time-domain view of the 1D vibration signal."""
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(t, signal, linewidth=0.7, color="#0A84FF")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title("Raw Vibration Signal", fontsize=11, fontweight="600")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_base64(fig)


def plot_spectrogram_from_signal(
    signal: np.ndarray,
    fs: int = 12000,
    nperseg: int = 256,
    noverlap: int = 128,
) -> str:
    """
    Visualise the STFT spectrogram using the same parameters as
    stft2spectrogram() — but renders a labelled matplotlib figure
    instead of returning a PIL image tensor.

    Pipeline mirrors stft2spectrogram:
      1. normalise signal
      2. scipy.signal.stft  (hann window, nperseg=256, noverlap=128)
      3. log1p(|Zxx|) then min-max normalise → plotted as a heatmap
    """
    from scipy.signal import stft

    # Use one representative segment (same length as training segments).
    # Taking the first 4096 samples keeps CWT fast and representative.
    chunk = signal[:min(len(signal), fs)]          # up to 1 second
    chunk = (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-8)

    f, t, Zxx = stft(chunk, fs=fs, window="hann",
                     nperseg=nperseg, noverlap=noverlap)
    spec = np.log1p(np.abs(Zxx))
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    pcm = ax.pcolormesh(t, f / 1000, spec, shading="auto", cmap="inferno")
    fig.colorbar(pcm, ax=ax, label="log1p |STFT| (norm)", pad=0.02, fraction=0.046)
    ax.set_ylabel("Frequency (kHz)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title("Spectrogram  (STFT)", fontsize=11, fontweight="600")
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_scalogram_from_signal(
    signal: np.ndarray,
    fs: int = 12000,
    wavelet: str = "morl",
    scale_min: int = 1,
    scale_max: int = 128,
) -> str:
    """
    Visualise the CWT scalogram using the same parameters as
    cwt2scalogram() — but renders a labelled matplotlib figure
    instead of returning a PIL image tensor.

    Pipeline mirrors cwt2scalogram:
      1. normalise signal
      2. pywt.cwt  (morl wavelet, scales 1..128, sampling_period=1/fs)
      3. log1p(|coeffs|) then min-max normalise → plotted as a heatmap
    """
    import pywt

    # Limit to 4096 samples — CWT is O(N*S) so keep it fast for viz
    chunk = signal[:min(len(signal), 4096)]
    chunk = (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-8)

    scales = np.arange(scale_min, scale_max + 1)
    coeffs, freqs = pywt.cwt(chunk, scales, wavelet, sampling_period=1 / fs)

    scalo = np.log1p(np.abs(coeffs))
    scalo = (scalo - scalo.min()) / (scalo.max() - scalo.min() + 1e-8)

    times = np.arange(len(chunk)) / fs

    fig, ax = plt.subplots(figsize=(5, 3.8))
    pcm = ax.pcolormesh(times, freqs / 1000, scalo, shading="auto", cmap="plasma")
    fig.colorbar(pcm, ax=ax, label="log1p |CWT| (norm)", pad=0.02, fraction=0.046)
    ax.set_ylabel("Frequency (kHz)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title("Scalogram  (CWT)", fontsize=11, fontweight="600")
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
def _load_signal_images(
    signal_path: str,
    transform_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run CWRU loader + transform. Returns (imgs[N,...], gts[N])."""
    transform_fn = TRANSFORM_FN[transform_key]
    normal = "Normal" in signal_path
    imgs, gts = cwru_inference(signal_path, transform_fn, normal=normal)
    return imgs, gts


def _load_raw_signal(signal_path: str) -> np.ndarray:
    """
    Load a 1D vibration segment from a .mat file for plotting.
    Keeps things simple: pick the first DE_time signal if present.
    """
    from scipy.io import loadmat
    mat = loadmat(signal_path)
    # CWRU files usually have a key like 'X097_DE_time'. Grab the first DE/FE.
    for k in mat:
        if k.startswith("__"):
            continue
        if "DE_time" in k or "FE_time" in k or "BA_time" in k:
            return np.asarray(mat[k]).squeeze()
    # Fallback: first numeric non-meta array
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v).squeeze()
        if arr.ndim == 1 and arr.size > 100:
            return arr
    raise ValueError(f"No 1D vibration signal found in {signal_path}")


# --------------------------------------------------------------------------- #
# Single-model inference (kept for backward compat with /predict)              #
# --------------------------------------------------------------------------- #
def fault_diagnosis(
    model_path: str,
    signal_path: str,
    transforms: str,
) -> Tuple[int, float, int, float, str, str]:
    """
    Run one ONNX model on one random segment.

    Returns:
        label, confidence, ground_truth, inference_time_ms, raw_plot_b64, transform_plot_b64
    """
    if transforms not in TRANSFORM_FN:
        raise ValueError("transforms must be 'cwt'/'scalogram' or 'stft'/'spectrogram'")

    # Normalize transform key
    transform_key = "scalogram" if transforms in ("cwt", "scalogram") else "spectrogram"

    imgs, gts = _load_signal_images(signal_path, transform_key)
    imgs = (imgs / 255.0).astype(np.float32)

    idx = random.randint(0, len(imgs) - 1)
    x = imgs[idx:idx + 1]
    gt = _encode_gt(gts[idx])

    model = ONNXModel(model_path)
    t0 = time.perf_counter()
    logits = model.predict(x)
    inference_time_ms = (time.perf_counter() - t0) * 1000.0

    prob = softmax(logits, axis=-1).squeeze()
    label = int(np.argmax(prob))
    confidence = float(np.max(prob))

    # Plots
    raw = _load_raw_signal(signal_path)
    raw_plot = plot_raw_signal(raw)
    transform_plot = (
        plot_scalogram_from_signal(raw)
        if transform_key == "scalogram"
        else plot_spectrogram_from_signal(raw)
    )

    return label, confidence, gt, inference_time_ms, raw_plot, transform_plot


# --------------------------------------------------------------------------- #
# Multi-model (8 ONNX) inference                                               #
# --------------------------------------------------------------------------- #
# Cache ONNXModel objects so we don't re-load weights on every request.
_MODEL_CACHE: Dict[str, ONNXModel] = {}


def _get_model(path: str) -> ONNXModel:
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = ONNXModel(path)
    return _MODEL_CACHE[path]


def _infer_one(
    arch: str,
    transform_key: str,
    x: np.ndarray,
    gt: int,
) -> Dict:
    """Run one (architecture, transform) combination and return a result dict."""
    model_path = MODEL_REGISTRY[arch][transform_key]
    try:
        if not os.path.exists(model_path):
            return {
                "model_name": arch,
                "transform": transform_key,
                "label": -1,
                "ground_truth": gt,
                "confidence": 0.0,
                "inference_time": 0.0,
                "correct": False,
                "error": f"Model file not found: {model_path}",
            }

        model = _get_model(model_path)
        t0 = time.perf_counter()
        logits = model.predict(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        prob = softmax(logits, axis=-1).squeeze()
        label = int(np.argmax(prob))
        confidence = float(np.max(prob))

        return {
            "model_name": arch,
            "transform": transform_key,
            "label": label,
            "ground_truth": gt,
            "confidence": confidence,
            "inference_time": elapsed_ms,
            "correct": label == gt,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "model_name": arch,
            "transform": transform_key,
            "label": -1,
            "ground_truth": gt,
            "confidence": 0.0,
            "inference_time": 0.0,
            "correct": False,
            "error": str(e),
        }


def fault_diagnosis_all(signal_path: str) -> Dict:
    """
    Non-streaming variant: run every model, return everything at once.
    Used by the /predict-all fallback endpoint.
    """
    # Load both representations (they pick independent segments below is NOT ok —
    # we want ONE segment index for fair comparison, so pick idx on scalogram set
    # and reuse on spectrogram after a parallel load).
    scal_imgs, scal_gts = _load_signal_images(signal_path, "scalogram")
    spec_imgs, spec_gts = _load_signal_images(signal_path, "spectrogram")

    n = min(len(scal_imgs), len(spec_imgs))
    idx = random.randint(0, n - 1)

    scal_imgs = (scal_imgs / 255.0).astype(np.float32)
    spec_imgs = (spec_imgs / 255.0).astype(np.float32)

    gt = _encode_gt(scal_gts[idx])

    raw = _load_raw_signal(signal_path)
    raw_plot = plot_raw_signal(raw)
    scalogram_plot = plot_scalogram_from_signal(raw)
    spectrogram_plot = plot_spectrogram_from_signal(raw)

    results: List[Dict] = []
    for arch in MODEL_REGISTRY:
        results.append(_infer_one(arch, "scalogram", scal_imgs[idx:idx + 1], gt))
        results.append(_infer_one(arch, "spectrogram", spec_imgs[idx:idx + 1], gt))

    return {
        "segment_idx": idx,
        "raw_plot": raw_plot,
        "spectrogram_plot": spectrogram_plot,
        "scalogram_plot": scalogram_plot,
        "results": results,
    }


# --------------------------------------------------------------------------- #
# Streaming generator — yields SSE-friendly dicts                              #
# --------------------------------------------------------------------------- #
def fault_diagnosis_stream(signal_path: str) -> Generator[Dict, None, None]:
    """
    Yields SSE events in order:
      status → raw → status → scalogram → status → spectrogram →
      meta → result ×8 → done
    Each step is wrapped individually so errors are logged with full
    tracebacks to the uvicorn console AND sent to the client.
    """

    def _safe_yield_plot(event_name: str, plot_fn, *args, **kwargs):
        """Call plot_fn, yield the plot event, or yield an error event."""
        try:
            plot = plot_fn(*args, **kwargs)
            return {"event": event_name, "data": {"plot": plot}}
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[stream] %s plot failed: %s", event_name, tb)
            return {"event": "error", "data": {
                "message": f"{event_name} plot failed: {exc}",
                "traceback": tb,
            }}

    # ── 1. Raw signal ────────────────────────────────────────────────────────
    yield {"event": "status", "data": {"message": "Loading .mat file..."}}
    try:
        raw = _load_raw_signal(signal_path)
        logger.info("[stream] raw signal loaded: %d samples", len(raw))
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[stream] _load_raw_signal failed: %s", tb)
        yield {"event": "error", "data": {"message": f"Cannot load signal: {exc}", "traceback": tb}}
        return

    yield {"event": "status", "data": {"message": "Plotting raw signal..."}}
    yield _safe_yield_plot("raw", plot_raw_signal, raw)

    # ── 2. Scalogram viz (pywt) ──────────────────────────────────────────────
    yield {"event": "status", "data": {"message": "Computing scalogram (CWT)..."}}
    yield _safe_yield_plot("scalogram", plot_scalogram_from_signal, raw)

    # ── 3. Spectrogram viz (STFT) ────────────────────────────────────────────
    yield {"event": "status", "data": {"message": "Computing spectrogram (STFT)..."}}
    yield _safe_yield_plot("spectrogram", plot_spectrogram_from_signal, raw)

    # ── 4. Load model-input images (cwru_inference + transforms) ────────────
    yield {"event": "status", "data": {"message": "Loading scalogram dataset..."}}
    try:
        scal_imgs, scal_gts = _load_signal_images(signal_path, "scalogram")
        logger.info("[stream] scalogram imgs: %s", str(scal_imgs.shape))
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[stream] scalogram _load_signal_images failed: %s", tb)
        yield {"event": "error", "data": {"message": f"Scalogram dataset failed: {exc}", "traceback": tb}}
        return

    yield {"event": "status", "data": {"message": "Loading spectrogram dataset..."}}
    try:
        spec_imgs, spec_gts = _load_signal_images(signal_path, "spectrogram")
        logger.info("[stream] spectrogram imgs: %s", str(spec_imgs.shape))
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[stream] spectrogram _load_signal_images failed: %s", tb)
        yield {"event": "error", "data": {"message": f"Spectrogram dataset failed: {exc}", "traceback": tb}}
        return

    n   = min(len(scal_imgs), len(spec_imgs))
    idx = random.randint(0, n - 1)
    gt  = _encode_gt(scal_gts[idx])
    logger.info("[stream] selected segment idx=%d  gt=%d  n=%d", idx, gt, n)

    scal_imgs = (scal_imgs / 255.0).astype(np.float32)
    spec_imgs = (spec_imgs / 255.0).astype(np.float32)

    yield {"event": "meta",
           "data": {"segment_idx": idx, "ground_truth": gt,
                    "total_models": len(MODEL_REGISTRY) * 2}}

    # ── 5. Run all 8 models ──────────────────────────────────────────────────
    x_scal = scal_imgs[idx:idx + 1]
    x_spec = spec_imgs[idx:idx + 1]

    for arch in MODEL_REGISTRY:
        yield {"event": "status",
               "data": {"message": f"Running {arch} · scalogram..."}}
        result = _infer_one(arch, "scalogram", x_scal, gt)
        logger.info("[stream] %s/scalogram → label=%s err=%s",
                    arch, result.get("label"), result.get("error"))
        yield {"event": "result", "data": result}

        yield {"event": "status",
               "data": {"message": f"Running {arch} · spectrogram..."}}
        result = _infer_one(arch, "spectrogram", x_spec, gt)
        logger.info("[stream] %s/spectrogram → label=%s err=%s",
                    arch, result.get("label"), result.get("error"))
        yield {"event": "result", "data": result}

    yield {"event": "done", "data": {}}