<div align="center">

# Bearing Fault Detection via GAN-Augmented Time-Frequency Imaging

**WGAN-GP · CWT · STFT · MobileNetV4 · GhostNetV3 · TinyNetD · EdgeNeXtXXS · ONNX INT8**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-INT8-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

</div>

---

## Overview

This project addresses the **class-imbalance problem** in the [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter) by using **Wasserstein GAN with Gradient Penalty (WGAN-GP)** to generate synthetic training images. Raw vibration signals are converted to either **Scalograms (CWT)** or **Spectrograms (STFT)**, and the augmented data is used to train four lightweight CNN classifiers. All models are exported to **ONNX INT8** format and served through a real-time **sliding-window monitor** with a web-based UI.

---

## Pipeline

```
Raw Vibration Signal (CWRU .mat)
          │
          ▼
  ┌───────────────────────────────────────┐
  │         Signal Preprocessing         │
  │                                       │
  │  CWT → Scalogram (128×128)            │
  │  STFT → Spectrogram (128×128)         │
  └───────────────────┬───────────────────┘
                      │
                      ▼
  ┌───────────────────────────────────────┐
  │           WGAN-GP Training            │
  │                                       │
  │  Generator ↔ Critic (λ_GP = 10)      │
  │  Critic iters per G step: 5           │
  │  Evaluated with FID & KID             │
  └───────────────────┬───────────────────┘
                      │ Synthetic images per class
                      ▼
  ┌───────────────────────────────────────┐
  │       Augmented Dataset               │
  │                                       │
  │  Normal · Inner Race · Ball Fault     │
  │  (Real + Synthetic, balanced)         │
  └───────────────────┬───────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    Scalogram     Spectrogram    (×4 models each)
          │           │
          └─────┬─────┘
                ▼
  ┌───────────────────────────────────────┐
  │         CNN Classifier Training       │
  │                                       │
  │  MobileNetV4  │  GhostNetV3           │
  │  TinyNetD     │  EdgeNeXtXXS          │
  │                                       │
  │  Metrics: Accuracy, F1, AUC-ROC,      │
  │           Cohen's κ, MCC              │
  └───────────────────┬───────────────────┘
                      │
                      ▼
  ┌───────────────────────────────────────┐
  │          ONNX INT8 Export             │
  │   8 model variants (4 arch × 2 prep)  │
  └───────────────────┬───────────────────┘
                      │
                      ▼
  ┌───────────────────────────────────────┐
  │         Real-Time Monitor             │
  │   FastAPI · SSE · Web UI              │
  │   ~58ms preproc · ~1.4ms inference    │
  └───────────────────────────────────────┘
```

---

## Dataset — CWRU Bearing

| Class | Label | Description |
|-------|-------|-------------|
| Normal | 0 | Healthy bearing |
| Outer Race Fault | 1 | Defect on outer raceway |
| Inner Race Fault | 2 | Defect on inner raceway |
| Ball Fault | 3 | Rolling element defect |

- Sampling rate: **12,000 Hz**
- Window size: **2,048 samples** (≈ 171 ms per segment)
- Each window is converted to a **128 × 128** grayscale image

---

## Signal-to-Image Transforms

### Scalogram — Continuous Wavelet Transform (CWT)

```
Signal → z-score normalize → pywt.cwt (Morlet, scales 1–128)
       → log1p magnitude → min-max normalize → 128×128 image
```

### Spectrogram — Short-Time Fourier Transform (STFT)

```
Signal → z-score normalize → scipy.signal.stft (Hann, nperseg=256, noverlap=128)
       → log1p magnitude → min-max normalize → 128×128 image
```

---

## GAN — WGAN-GP

| Hyperparameter | Value |
|----------------|-------|
| Gradient penalty weight (λ) | 10 |
| Critic iterations per generator step | 5 |
| GAN evaluation metric | FID, KID |
| Output | Synthetic scalogram / spectrogram images |

Training GIFs are saved automatically:

| Class | GIF |
|-------|-----|
| Normal | `gif/N_2000_epochs.gif` |
| Inner Race Fault | `gif/IR_700_epochs.gif` |
| Ball Fault | `gif/B_700_epochs.gif` |

---

## CNN Models

Eight model variants are produced — each of the four architectures trained on both preprocessing types:

| Architecture | Preprocessing | Format |
|---|---|---|
| MobileNetV4 | Scalogram | ONNX INT8 |
| MobileNetV4 | Spectrogram | ONNX INT8 |
| GhostNetV3 | Scalogram | ONNX INT8 |
| GhostNetV3 | Spectrogram | ONNX INT8 |
| TinyNetD | Scalogram | ONNX INT8 |
| TinyNetD | Spectrogram | ONNX INT8 |
| EdgeNeXtXXS | Scalogram | ONNX INT8 |
| EdgeNeXtXXS | Spectrogram | ONNX INT8 |

### Training Features

- Mixed-precision training (AMP) with gradient clipping
- Early stopping (patience-based)
- LR scheduler support (`ReduceLROnPlateau` / step-based)
- Full academic evaluation suite:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Precision / Recall / F1 | Macro-averaged |
| AUC-ROC | One-vs-rest, macro |
| Cohen's κ | Agreement beyond chance |
| MCC | Matthews Correlation Coefficient |

---

## Real-Time Monitor

The monitor streams sliding-window predictions over a MATLAB `.mat` signal file, displaying results live in the browser.

**Launch:**

```bash
uvicorn monitor.app.main:app --reload --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### UI Features

| Panel | Description |
|-------|-------------|
| Raw Signal | Time-domain waveform plot |
| Prediction Bar | Per-window predicted class, color-coded |
| Ground Truth Bar | True labels (when available in `.mat`) |
| Session Summary | Normal count, Fault count, Avg preproc time, Avg model time |
| Prediction History | Scrollable log — timestamp, label, confidence %, preproc ms, model ms |

### Inference Settings

| Setting | Options |
|---------|---------|
| Transform | Scalogram (CWT) · Spectrogram (STFT) |
| Model | MobileNetV4 · GhostNetV3 · TinyNetD · EdgeNeXtXXS |
| Confidence Threshold | 0 – 100 % (default 80%) — low-confidence windows fall back to *Normal* |
| Ground Truth Bar | Toggle visibility |

### Inference Performance (approximate)

| Stage | Time |
|-------|------|
| Preprocessing (CWT/STFT) | ~58 ms |
| Model inference (ONNX INT8) | ~1.4 ms |

---

## Project Structure

```
Capstone/
├── src/
│   ├── data/               # CWRU loader, dataset classes, test-case generator
│   ├── features/
│   │   ├── scalogram.py    # CWT → 128×128 image
│   │   └── spectrogram.py  # STFT → 128×128 image
│   ├── models/
│   │   ├── architecture/   # WGAN-GP Generator & Discriminator, CNN backbones
│   │   └── onnx/           # ONNX inference wrapper
│   ├── training/
│   │   ├── trainer_wgan_gp.py      # WGAN-GP trainer (FID/KID eval, GIF export)
│   │   └── trainer_classifier.py   # CNN trainer (AMP, early stopping, plots)
│   ├── evaluation/         # FID, KID, sklearn metrics
│   └── utils/
├── monitor/
│   └── app/
│       ├── main.py         # FastAPI app entry point
│       ├── router.py       # SSE stream, sliding-window inference
│       └── static/         # index.html + monitor.js (web UI)
├── dataset/
│   ├── processed/          # Scalogram & Spectrogram .pt tensors (train/val/test)
│   └── generated/          # WGAN-GP synthetic .pt tensors per class
├── models/
│   └── ONNX/CNN/           # Exported ONNX INT8 models (SCALOGRAM / SPECTROGRAM)
├── notebooks/              # Experiment notebooks
├── gif/                    # WGAN-GP training progress GIFs
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version |
|---------|---------|
| fastapi | 0.121.1 |
| uvicorn | latest |
| onnxruntime | 1.24.4 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| pywavelets | 1.7.0 |
| pydantic | 2.12.4 |
| pillow | 12.1.1 |

---

## Notebooks

Jupyter notebooks covering the full workflow — preprocessing, WGAN-GP training, CNN training, ONNX export — are available in the [`notebooks/`](notebooks/) directory.

---

<div align="center">

*Capstone Project — Bearing Fault Detection with GAN-Based Data Augmentation*

</div>
