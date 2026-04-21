"""
Monitor router — SSE stream that slides a window across the signal,
runs the chosen model on each window, and streams results back.
"""
import json
import logging
import time
import traceback

import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy.special import softmax
from scipy.io import loadmat

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────
BASE = r"D:\Capstone\models\ONNX\CNN"
FS   = 12000
WIN  = 2048
HOP  = 2048   # non-overlapping — matches cwru_inference overlap=0

USE_INT8 = False
_SUFFIX  = "best_model_int8.onnx" if USE_INT8 else "best_model_fp32.onnx"

MODEL_PATHS = {
    "MobileNetV4": {
        "scalogram":   rf"{BASE}\SCALOGRAM\MobileNetV4\{_SUFFIX}",
        "spectrogram": rf"{BASE}\SPECTROGRAM\MobileNetV4\{_SUFFIX}",
    },
    "TinyNetD": {
        "scalogram":   rf"{BASE}\SCALOGRAM\TinyNetD\{_SUFFIX}",
        "spectrogram": rf"{BASE}\SPECTROGRAM\TinyNetD\{_SUFFIX}",
    },
    "EdgeNeXtXXS": {
        "scalogram":   rf"{BASE}\SCALOGRAM\EdgenextXXS\{_SUFFIX}",
        "spectrogram": rf"{BASE}\SPECTROGRAM\EdgenextXXS\{_SUFFIX}",
    },
    "GhostNetV3": {
        "scalogram":   rf"{BASE}\SCALOGRAM\GhostNetV3\{_SUFFIX}",
        "spectrogram": rf"{BASE}\SPECTROGRAM\GhostNetV3\{_SUFFIX}",
    },
}

LABEL_NAMES = {0: "Normal", 1: "Outer Race Fault", 2: "Inner Race Fault", 3: "Ball Fault"}
LABEL_SHORT = {0: "Normal",  1: "OR Fault",         2: "IR Fault",         3: "Ball Fault"}

# ── Model cache ───────────────────────────────────────────────────────────────
from src.models.onnx_inference import ONNXModel
_CACHE: dict = {}

def _get_model(path: str) -> ONNXModel:
    if path not in _CACHE:
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        logger.info("[model] Loading: %s", path)
        _CACHE[path] = ONNXModel(path)
    return _CACHE[path]

# ── Transform — split preprocessing vs model time ─────────────────────────────
def _to_image(window: np.ndarray, transform: str):
    """
    Returns (x, preproc_ms) where x is (1,1,128,128) float32 tensor.
    Preprocessing time includes CWT/STFT computation only.
    """
    t0 = time.perf_counter()
    if transform == "scalogram":
        from src.features.cwt import cwt2scalogram
        img = cwt2scalogram(window, img_size=128, gray=True)
    else:
        from src.features.stft import stft2spectrogram
        img = stft2spectrogram(window, img_size=128, gray=True)
    preproc_ms = (time.perf_counter() - t0) * 1000.0
    x = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    return x, preproc_ms

# ── Signal + ground truth loading ─────────────────────────────────────────────
def _load_mat(path: str):
    """
    Returns (signal, ground_truth_or_None).
    - Signal:       key containing '_DE_time' but NOT 'groud_truth'
    - Ground truth: key containing 'groud_truth' (sample-level int array)
    """
    mat = loadmat(path)
    sig = None
    gt  = None
    for k in mat:
        if k.startswith("__"):
            continue
        if "groud_truth" in k or "ground_truth" in k:
            gt = np.asarray(mat[k]).squeeze().astype(int)
        elif "DE_time" in k:
            sig = np.asarray(mat[k]).squeeze()
    if sig is None:
        keys = [k for k in mat if not k.startswith("__")]
        raise ValueError(f"No DE_time key found in {path}. Keys: {keys}")
    return sig, gt

# ── Ground truth → window labels ─────────────────────────────────────────────
def _gt_window_labels(gt_samples: np.ndarray, total_windows: int) -> list:
    """
    Derive one ground truth label per window by majority vote over the
    WIN samples that window covers.
    gt_samples: int array, same length as signal, values 0–3.
    """
    labels = []
    for i in range(total_windows):
        start = i * HOP
        end   = start + WIN
        chunk = gt_samples[start:min(end, len(gt_samples))]
        if len(chunk) == 0:
            labels.append(0)
        else:
            counts = np.bincount(chunk, minlength=4)
            labels.append(int(np.argmax(counts)))
    return labels

# ── SSE helper ────────────────────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("\n", " ")
    return f"event: {event}\ndata: {payload}\n\n"

# ── Request schema ────────────────────────────────────────────────────────────
class MonitorRequest(BaseModel):
    signal_path: str
    model_name:  str
    transform:   str
    threshold:   float = 80.0   # confidence threshold %, default 80

# ── Signal metadata endpoint ──────────────────────────────────────────────────
@router.post("/signal-info")
def signal_info(req: MonitorRequest):
    sig, gt = _load_mat(req.signal_path)
    if "Normal" in req.signal_path:
        from scipy.signal import resample_poly
        sig = resample_poly(sig, up=1, down=4)
        if gt is not None:
            gt = resample_poly(gt.astype(float), up=1, down=4)
            gt = np.round(gt).astype(int)

    total_windows = max(0, (len(sig) - WIN) // HOP + 1)
    duration      = len(sig) / FS

    target = 2000
    step   = max(1, len(sig) // target)
    ds     = sig[::step].tolist()

    # Pre-compute ground truth window labels if available
    gt_window_labels = None
    if gt is not None:
        gt_window_labels = _gt_window_labels(gt, total_windows)

    return {
        "n_samples":         int(len(sig)),
        "duration_s":        round(duration, 3),
        "total_windows":     int(total_windows),
        "fs":                FS,
        "win":               WIN,
        "hop":               HOP,
        "waveform":          ds,
        "waveform_step":     int(step),
        "has_ground_truth":  gt is not None,
        "gt_window_labels":  gt_window_labels,   # list[int] or null
    }

# ── Main SSE stream ───────────────────────────────────────────────────────────
@router.post("/monitor-stream")
def monitor_stream(req: MonitorRequest):
    def generate():
        try:
            sig, gt = _load_mat(req.signal_path)
            if "Normal" in req.signal_path:
                from scipy.signal import resample_poly
                sig = resample_poly(sig, up=1, down=4)
                if gt is not None:
                    gt_arr = resample_poly(gt.astype(float), up=1, down=4)
                    gt = np.round(gt_arr).astype(int)

            n             = len(sig)
            total_windows = max(0, (n - WIN) // HOP + 1)
            gt_labels     = _gt_window_labels(gt, total_windows) if gt is not None else None

            model = _get_model(MODEL_PATHS[req.model_name][req.transform])
            threshold = float(req.threshold)  # e.g. 80.0

            yield _sse("meta", {
                "total_windows":    total_windows,
                "n_samples":        n,
                "duration_s":       round(n / FS, 3),
                "has_ground_truth": gt is not None,
            })

            for i in range(total_windows):
                start    = i * HOP
                end      = start + WIN
                window   = sig[start:end].copy()
                t_center = (start + WIN / 2) / FS

                # ── Preprocessing (CWT / STFT) ────────────────────────────
                x, preproc_ms = _to_image(window, req.transform)
                logger.info("[infer] window[%d] shape=%s min=%.4f max=%.4f mean=%.4f",
                            i, x.shape, float(x.min()), float(x.max()), float(x.mean()))

                # ── Model inference ───────────────────────────────────────
                t1       = time.perf_counter()
                logits   = model.predict(x)
                model_ms = (time.perf_counter() - t1) * 1000.0

                prob       = softmax(logits, axis=1).squeeze()
                raw_label  = int(np.argmax(prob))
                confidence = float(np.max(prob)) * 100.0

                # ── Confidence threshold → fallback to Normal ─────────────
                label = raw_label if confidence >= threshold else 0

                # ── Ground truth for this window ──────────────────────────
                gt_label = gt_labels[i] if gt_labels is not None else None

                yield _sse("window", {
                    "window_idx":   i,
                    "total":        total_windows,
                    "start_sample": start,
                    "end_sample":   end,
                    "t_start":      round(start / FS, 4),
                    "t_center":     round(t_center, 4),
                    "t_end":        round(end / FS, 4),
                    "label":        label,
                    "raw_label":    raw_label,
                    "label_name":   LABEL_NAMES.get(label, f"Class {label}"),
                    "label_short":  LABEL_SHORT.get(label, f"Class {label}"),
                    "confidence":   round(confidence, 1),
                    "below_threshold": confidence < threshold,
                    "preproc_ms":   round(preproc_ms, 1),
                    "model_ms":     round(model_ms, 1),
                    "inf_ms":       round(preproc_ms + model_ms, 1),
                    "is_fault":     label != 0,
                    "gt_label":     gt_label,
                    "gt_name":      LABEL_NAMES.get(gt_label, "Unknown") if gt_label is not None else None,
                })

                window_dur = WIN / FS
                sleep_s    = max(0.0, window_dur - (preproc_ms + model_ms) / 1000.0)
                time.sleep(sleep_s)

            yield _sse("done", {"total_windows": total_windows})

        except Exception:
            tb = traceback.format_exc()
            logger.error("[monitor-stream] %s", tb)
            yield _sse("error", {"message": tb.splitlines()[-1]})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )