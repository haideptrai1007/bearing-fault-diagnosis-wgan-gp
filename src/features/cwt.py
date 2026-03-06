import numpy as np
import pywt
from PIL import Image

def cwt2scalogram(signal, img_size=128, gray=False, fs=12000, wavelet='morl', scale_min=1, scale_max=128):
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    scales = np.arange(scale_min, scale_max + 1)

    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

    scalogram = np.log1p(np.abs(coeffs))
    scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min() + 1e-8)

    image = Image.fromarray((scalogram * 255).astype(np.uint8), mode='L')
    image = image.resize((img_size, img_size), Image.BILINEAR)

    if gray:
        image = np.array(image, dtype=np.float32) / 255.0
        image = image[None, :, :]
    else:
        image = np.stack([np.array(image)] * 3, axis=-1) / 255.0
        image = np.transpose(image, (2, 0, 1))

    return image