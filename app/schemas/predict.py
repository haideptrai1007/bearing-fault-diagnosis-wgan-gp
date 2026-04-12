from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    model_path: str
    signal_path: str
    transforms: str

class PredictResponse(BaseModel):
    label: int
    confidence: float