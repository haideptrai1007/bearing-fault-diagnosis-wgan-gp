import numpy as np
from scipy.signal import stft
from PIL import Image

def stft2spectrogram(signal, img_size=128, gray=False, fs=12000, window="hann", nperseg=256, noverlap=128):
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    spectrogram = np.log1p(np.abs(Zxx))

    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)

    image = Image.fromarray((spectrogram * 255).astype(np.uint8), mode='L')
    image = image.resize((img_size, img_size), Image.BILINEAR)
    
    if gray:
        image = np.array(image, dtype=np.float32) / 255.0
        image = image[None, :, :]
    else:
        image = np.stack([np.array(image)] * 3, axis=-1) / 255.0
        image = np.transpose(image, (2, 0, 1))

    return image