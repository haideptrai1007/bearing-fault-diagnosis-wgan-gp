import numpy as np
from src.data.cwru_data import cwru_inference
from src.features.cwt import cwt2scalogram
from src.features.stft import stft2spectrogram
from src.models.onnx_inference import ONNXModel
import random
from scipy.special import softmax

def fault_diagnosis(model_path, signal_path, transforms=None):

    model = ONNXModel(model_path)

    transform_map = {
        "cwt": cwt2scalogram,
        "stft": stft2spectrogram
    }
    
    if transforms not in transform_map:
        raise ValueError("transforms must be 'cwt' or 'stft'")
    
    transform_fn = transform_map[transforms]
    
    if "Normal" in signal_path:
        imgs = cwru_inference(signal_path, transform_fn, normal=True)
    else:
        imgs = cwru_inference(signal_path, transform_fn, normal=False)

    imgs = (imgs / 255.0).astype(np.float32)

    idx = random.randint(0, len(imgs) - 1)

    x = imgs[idx:idx+1]

    logits = model.predict(x)
    prob = softmax(logits)

    label = int(np.argmax(prob))
    confidence = float(np.max(prob))

    return label, confidence