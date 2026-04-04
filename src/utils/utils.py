import numpy as np
import torch

def sliding_window(signal, window_size=2048, overlap=0.5):
    stride = int(window_size * (1 - overlap))
    windows = []
    for i in range(0, len(signal) - window_size + 1, stride):
        window = signal[i:i + window_size]
        windows.append(window)
    return np.array(windows)


def save_data(data, save_path):
    tensor = torch.from_numpy(data['data'])
    labels = torch.from_numpy(data['label'])
    dict = {
        'data': tensor,
        'label': labels
    }
    torch.save(dict, save_path)