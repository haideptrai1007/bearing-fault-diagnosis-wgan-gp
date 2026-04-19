from typing import Dict, List, Optional
from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Single model inference request."""
    model_path: str
    signal_path: str
    transforms: str  # 'cwt' or 'stft'


class PredictResponse(BaseModel):
    """Single model inference response."""
    label: int
    ground_truth: int
    confidence: float
    inference_time: float
    raw_plot: str
    transform_plot: str


class PlotRequest(BaseModel):
    """Request for just the plots (no inference)."""
    signal_path: str


class PlotResponse(BaseModel):
    """Raw signal + spectrogram + scalogram preview plots."""
    raw_plot: str          # base64 PNG
    spectrogram_plot: str  # base64 PNG
    scalogram_plot: str    # base64 PNG
    segment_idx: int       # which segment was picked


class PredictAllRequest(BaseModel):
    """Run all 8 models (4 architectures × 2 transforms) on one signal."""
    signal_path: str


class ModelResult(BaseModel):
    """Single model result inside PredictAllResponse."""
    model_name: str        # e.g. 'MobileNetV4'
    transform: str         # 'scalogram' or 'spectrogram'
    label: int
    ground_truth: int
    confidence: float
    inference_time: float  # milliseconds
    correct: bool
    error: Optional[str] = None


class PredictAllResponse(BaseModel):
    """Response for /predict-all (non-streaming variant)."""
    segment_idx: int
    raw_plot: str
    spectrogram_plot: str
    scalogram_plot: str
    results: List[ModelResult]