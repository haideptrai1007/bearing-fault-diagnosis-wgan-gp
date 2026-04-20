from typing import List, Optional
from pydantic import BaseModel


class PredictAllRequest(BaseModel):
    signal_path: str


class ModelResult(BaseModel):
    model_name: str
    transform: str
    label: int
    ground_truth: int
    confidence: float
    inference_time: float
    correct: bool
    error: Optional[str] = None


class PredictAllResponse(BaseModel):
    segment_idx: int
    ground_truth: int
    plot_time_ms: float
    raw_plot: str
    spectrogram_plot: str
    scalogram_plot: str
    results: List[ModelResult]