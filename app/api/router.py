"""
API router — single POST /predict-all, no streaming.
"""
import logging
import traceback

from fastapi import APIRouter, HTTPException
from app.schemas.predict import PredictAllRequest, PredictAllResponse, ModelResult
from app.services.inference import run_all

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict-all", response_model=PredictAllResponse)
def predict_all(request: PredictAllRequest):
    try:
        out = run_all(request.signal_path)
        return PredictAllResponse(
            segment_idx=out["segment_idx"],
            ground_truth=out["ground_truth"],
            plot_time_ms=out["plot_time_ms"],
            raw_plot=out["raw_plot"],
            spectrogram_plot=out["spectrogram_plot"],
            scalogram_plot=out["scalogram_plot"],
            results=[ModelResult(**r) for r in out["results"]],
        )
    except Exception as e:
        logger.error("[predict-all] %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))