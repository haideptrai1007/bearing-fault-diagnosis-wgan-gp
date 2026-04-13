# import numpy as np
# from src.data.cwru_data import cwru_inference
# from src.features.cwt import cwt2scalogram
# from src.features.stft import stft2spectrogram
# from src.models.onnx_inference import ONNXModel
# import random
# from scipy.special import softmax

# def fault_diagnosis(model_path, signal_path, transforms=None):

#     model = ONNXModel(model_path)

#     transform_map = {
#         "cwt": cwt2scalogram,
#         "stft": stft2spectrogram
#     }
    
#     if transforms not in transform_map:
#         raise ValueError("transforms must be 'cwt' or 'stft'")
    
#     transform_fn = transform_map[transforms]
    
#     if "Normal" in signal_path:
#         imgs = cwru_inference(signal_path, transform_fn, normal=True)
#     else:
#         imgs = cwru_inference(signal_path, transform_fn, normal=False)

#     imgs = (imgs / 255.0).astype(np.float32)

#     idx = random.randint(0, len(imgs) - 1)

#     x = imgs[idx:idx+1]

#     logits = model.predict(x)
#     prob = softmax(logits)

#     label = int(np.argmax(prob))
#     confidence = float(np.max(prob))

#     return label, confidence

import numpy as np
import random
import scipy.io
import base64
import io
from scipy.special import softmax
from scipy.signal import resample_poly
from src.data.cwru_data import cwru_inference, sliding_window
from src.features.cwt import cwt2scalogram
from src.features.stft import stft2spectrogram
from src.models.onnx_inference import ONNXModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WINDOW_SIZE = 2048
IMG_SIZE = 128

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0a0a0f', edgecolor='none', dpi=130)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def style_ax(ax, title):
    ax.set_facecolor('#111118')
    for spine in ax.spines.values():
        spine.set_edgecolor('#ffffff12')
    ax.tick_params(colors='#ffffff40', labelsize=8)
    ax.set_title(title, color='#ffffffbb', fontsize=10, pad=10)

def plot_raw_signal(signal, idx):
    start   = idx * WINDOW_SIZE
    segment = signal[start : start + WINDOW_SIZE]

    fig, ax = plt.subplots(figsize=(5, 2))
    fig.patch.set_facecolor('#0a0a0f')
    style_ax(ax, f'Raw signal — segment {idx}')
    ax.plot(segment, color='#fc3c44', linewidth=0.7, alpha=0.9)
    ax.set_xlabel('Sample', color='#ffffff55', fontsize=9)
    ax.set_ylabel('Amplitude', color='#ffffff55', fontsize=9)
    ax.set_xlim(0, WINDOW_SIZE)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)

def plot_transform_image(img):
    # img shape: (1, 128, 128) → squeeze to (128, 128)
    img_2d = img.squeeze()

    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor('#0a0a0f')
    style_ax(ax, 'Transformed image')
    ax.imshow(img_2d, cmap='inferno', aspect='auto', origin='lower')
    ax.set_xlabel('Time', color='#ffffff55', fontsize=9)
    ax.set_ylabel('Frequency', color='#ffffff55', fontsize=9)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)

def fault_diagnosis(model_path, signal_path, transforms=None):
    model = ONNXModel(model_path)

    transform_map = {
        "cwt":  cwt2scalogram,
        "stft": stft2spectrogram
    }

    if transforms not in transform_map:
        raise ValueError("transforms must be 'cwt' or 'stft'")

    transform_fn = transform_map[transforms]
    normal = "Normal" in signal_path

    # ── Load raw signal (same logic as cwru_inference) ──────────
    mat_file = scipy.io.loadmat(signal_path)
    signal = None
    for k, v in mat_file.items():
        if "DE" in k:
            signal = v.squeeze()
            break
    if signal is None:
        raise ValueError("No DE channel found in .mat file")

    if normal:
        signal = resample_poly(signal, up=1, down=4)

    # ── Run inference ────────────────────────────────────────────
    imgs = cwru_inference(signal_path, transform_fn,
                          window_size=WINDOW_SIZE, normal=normal, gray=True)
    imgs = (imgs / 255.0).astype(np.float32)

    idx = random.randint(0, len(imgs) - 1)
    x   = imgs[idx:idx+1]

    logits = model.predict(x)
    prob   = softmax(logits)

    label      = int(np.argmax(prob))
    confidence = float(np.max(prob))

    # ── Plots ────────────────────────────────────────────────────
    raw_plot       = plot_raw_signal(signal, idx)
    transform_plot = plot_transform_image(imgs[idx])

    return label, confidence, raw_plot, transform_plot