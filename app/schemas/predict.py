# from pydantic import BaseModel
# from typing import List

# class PredictRequest(BaseModel):
#     model_path: str
#     signal_path: str
#     transforms: str

# class PredictResponse(BaseModel):
#     label: int
#     confidence: float

from pydantic import BaseModel

class PredictRequest(BaseModel):
    model_path:  str
    signal_path: str
    transforms:  str

class PredictResponse(BaseModel):
    label:           int
    confidence:      float
    raw_plot:        str   # base64 png
    transform_plot:  str   # base64 png