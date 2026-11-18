"""PathSnap FastAPI backend (runs on laptop 1).

Start (from the repo root):
    python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

Environment variables
    PATHSNAP_MODEL       lightgbm (default) or mlp
    PATHSNAP_MODEL_DIR   artifacts folder. lightgbm: lightgbm_model.joblib + config.json (default <repo>/results).
                         mlp: pathsnap_mlp_weights.pt + feature_scaler.joblib + model_config.json (default <repo>/saved_models)
    PATHSNAP_DATA_DIR    folder with one sub-folder per trip: <trip>/<trip>.track|.nodes|.arcs|.route
                         (default: <repo>/server_data/evaluator_test)
"""
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.pipeline import create_pipeline

log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.propagate = False

MODEL_KIND, PIPE = create_pipeline()
log.info("model kind: %s", MODEL_KIND)
log.info("model dir : %s", Path(PIPE.model_dir).resolve())
log.info("data dir  : %s", Path(PIPE.data_dir).resolve())
log.info("model     : %s", PIPE.model_info()["name"])
log.info("trips     : %d available %s", len(PIPE.trip_ids()), PIPE.trip_ids())

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "?"
    response = await call_next(request)
    log.info("%s %s %s -> %d (%.0f ms)", client, request.method, request.url.path, response.status_code,
             1000 * (time.perf_counter() - t0))
    return response


@app.get("/")
def root():
    return {"service": "PathSnap API", "endpoints": ["GET /health", "GET /trips", "POST /predict", "GET /predict/{trip_id}"]}


@app.get("/health")
def health():
    return {"status": "ok", "model": PIPE.model_info(), "n_trips": len(PIPE.trip_ids())}


@app.get("/trips")
def trips():
    """Trip IDs for the UI. No labels, no reference routes."""
    return {"trips": [{"trip_id": t, "n_gps_points": PIPE.n_points(t)} for t in PIPE.trip_ids()]}


def _run(trip_id: str, client: str):
    if trip_id not in PIPE.trip_ids():
        log.warning("PREDICT rejected: unknown trip %r from %s", trip_id, client)
        raise HTTPException(status_code=404, detail=f"Unknown trip_id {trip_id!r}")
    log.info("PREDICT request  trip=%s from %s", trip_id, client)
    try:
        res = PIPE.predict(trip_id)
        metrics = PIPE.score(trip_id, res)        # server-side only, uses the hidden .route file
        geo = PIPE.geometry(res)
    except Exception as exc:                       # noqa: BLE001
        log.exception("PREDICT failed trip=%s", trip_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    tm = res["timings_s"]
    if metrics:
        log.info("PREDICT done     trip=%s points=%d | top1 HMM=%.4f clf-only=%.4f | continuity=%.4f seqF1=%.4f | "
                 "%.2fs (features %.2f, model %.2f, viterbi %.2f)", trip_id, len(geo["gps_points"]),
                 metrics["hmm_viterbi"]["top1_accuracy"], metrics["classifier_only"]["top1_accuracy"],
                 metrics["hmm_viterbi"]["route_continuity"], metrics["hmm_viterbi"]["sequence_f1"],
                 tm["total"], tm["features"], tm["model"], tm["viterbi"])
    return {
        "trip_id": trip_id,
        "n_gps_points": len(geo["gps_points"]),
        "metrics": metrics,
        "timings_s": {k: round(v, 4) for k, v in tm.items()},
        "model": PIPE.model_info()["name"],
        **geo,
    }


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    return _run(req.trip_id, request.client.host if request.client else "?")


@app.get("/predict/{trip_id}")
def predict_get(trip_id: str, request: Request):
    return _run(trip_id, request.client.host if request.client else "?")
