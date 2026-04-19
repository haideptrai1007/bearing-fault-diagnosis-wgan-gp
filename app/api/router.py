"""
API router.

Endpoints:
  POST /predict           -> single-model inference (legacy)
  POST /predict-all       -> all 8 models, non-streaming (fallback)
  POST /predict-stream    -> all 8 models, Server-Sent Events (main UI flow)
  GET  /stream-test       -> tiny SSE smoke-test (no ML, no files needed)
"""
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.predict import (
    ModelResult,
    PredictAllRequest,
    PredictAllResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.inference import (
    fault_diagnosis,
    fault_diagnosis_all,
    fault_diagnosis_stream,
)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _sse(event_dict: dict) -> str:
    """
    Serialise one event to the SSE wire format.

    We use the named-event form so the browser can tell events apart:

        event: raw
        data: {"plot": "data:image/png;base64,..."}

    This is more robust than embedding the event name inside the JSON,
    because the browser's built-in EventSource (and our manual fetch reader)
    can split on the blank line without touching the (potentially huge) data.
    """
    name    = event_dict["event"]
    payload = json.dumps(event_dict["data"], ensure_ascii=False)
    # Guard: payload must not contain a bare newline (base64 doesn't, but be safe)
    payload = payload.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return f"event: {name}\ndata: {payload}\n\n"


# --------------------------------------------------------------------------- #
# Smoke-test endpoint — no ML, no files, just verifies SSE works end-to-end   #
# --------------------------------------------------------------------------- #
@router.get("/stream-test")
def stream_test():
    """
    Hit this from the browser console to verify SSE parsing works:

        fetch('/stream-test').then(r => {
          const reader = r.body.getReader();
          const dec = new TextDecoder();
          (async () => {
            while (true) {
              const {value, done} = await reader.read();
              if (done) break;
              console.log(dec.decode(value));
            }
          })();
        });
    """
    def gen():
        for i in range(5):
            yield _sse({"event": "status", "data": {"message": f"step {i}"}})
            time.sleep(0.1)
        yield _sse({"event": "done", "data": {}})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# --------------------------------------------------------------------------- #
# Legacy single-model endpoint                                                 #
# --------------------------------------------------------------------------- #
@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    label, confidence, gt, inf_time, raw_plot, transform_plot = fault_diagnosis(
        request.model_path,
        request.signal_path,
        request.transforms,
    )
    return PredictResponse(
        label=label,
        ground_truth=gt,
        confidence=confidence,
        inference_time=inf_time,
        raw_plot=raw_plot,
        transform_plot=transform_plot,
    )


# --------------------------------------------------------------------------- #
# Non-streaming fallback                                                       #
# --------------------------------------------------------------------------- #
@router.post("/predict-all", response_model=PredictAllResponse)
def predict_all(request: PredictAllRequest):
    out = fault_diagnosis_all(request.signal_path)
    return PredictAllResponse(
        segment_idx=out["segment_idx"],
        raw_plot=out["raw_plot"],
        spectrogram_plot=out["spectrogram_plot"],
        scalogram_plot=out["scalogram_plot"],
        results=[ModelResult(**r) for r in out["results"]],
    )


# --------------------------------------------------------------------------- #
# Main streaming endpoint                                                      #
# --------------------------------------------------------------------------- #
@router.post("/predict-stream")
def predict_stream(request: PredictAllRequest):
    """
    Server-Sent Events stream using named events.

    Wire format per event:
        event: <name>\\n
        data: <json>\\n
        \\n

    The data line is guaranteed to have no embedded newlines (base64 is safe).
    """
    def event_generator():
        for ev in fault_diagnosis_stream(request.signal_path):
            yield _sse(ev)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )