# from fastapi import APIRouter
# from app.schemas.predict import PredictRequest, PredictResponse
# from app.services.inference import fault_diagnosis

# router = APIRouter()

# @router.post("/predict", response_model=PredictResponse)
# def predict(request: PredictRequest):
#     label, confidence = fault_diagnosis(
#         request.model_path,
#         request.signal_path,
#         request.transforms,
#     )

#     return PredictResponse(
#         label=label,
#         confidence=confidence
#     )

from fastapi import APIRouter
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.inference import fault_diagnosis

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    label, confidence, raw_plot, transform_plot = fault_diagnosis(
        request.model_path,
        request.signal_path,
        request.transforms,
    )
    return PredictResponse(
        label=label,
        confidence=confidence,
        raw_plot=raw_plot,
        transform_plot=transform_plot,
    )