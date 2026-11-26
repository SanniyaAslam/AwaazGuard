"""Model server: one instance per model variant (champion = Production, candidate = Staging).

Env vars:
  MODEL_VARIANT   champion | candidate               (label only)
  MODEL_STAGE     Production | Staging               (registry stage to load)
  MODEL_NAME      awaazguard-aasist
  MODEL_DIR       optional local folder with model.onnx + reference.npz (skips registry)
  MLFLOW_TRACKING_URI / USERNAME / PASSWORD          (DagsHub credentials)

Only needed with MODEL_BACKEND=http (one container per model on bigger machines).
The default free-tier setup embeds both models in the gateway instead.

Run locally:  uvicorn src.serving.inference_app:app --port 8001
"""
from __future__ import annotations

import os
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from starlette.responses import Response

from src.serving.audio import AudioError, decode_bytes, trim_silence
from src.serving.backends import INFER_SECONDS
from src.serving.backends import MODEL_VERSION as MODEL_LOADED
from src.serving.model_store import LoadedModel, load_from_dir, load_from_registry
from src.utils.common import get_logger

log = get_logger("inference")
VARIANT = os.getenv("MODEL_VARIANT", "champion")
STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_NAME = os.getenv("MODEL_NAME", "awaazguard-aasist")
MODEL_DIR = os.getenv("MODEL_DIR", "")

INFER_TOTAL = Counter("awaaz_inference_total", "Inference requests", ["variant", "status"])

app = FastAPI(title=f"AwaazGuard inference ({VARIANT})", version="1.0")
_model: LoadedModel | None = None
_lock = threading.Lock()


def load_model() -> LoadedModel | None:
    global _model
    with _lock:
        try:
            m = load_from_dir(MODEL_DIR) if MODEL_DIR else load_from_registry(MODEL_NAME, STAGE)
        except Exception as e:  # registry unreachable -> keep serving the old model
            log.error("Model load failed: %s", e)
            return _model
        if _model is not None:
            MODEL_LOADED.labels(VARIANT, _model.version).set(0)
        _model = m
        if m:
            MODEL_LOADED.labels(VARIANT, m.version).set(1)
            log.info("Loaded %s", m.source)
        else:
            log.warning("No model in stage %s - variant %s idle", STAGE, VARIANT)
        return _model


@app.on_event("startup")
def _startup() -> None:
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "variant": VARIANT, "model_loaded": _model is not None}


@app.get("/info")
def info():
    if _model is None:
        return {"variant": VARIANT, "stage": STAGE, "loaded": False}
    return {"variant": VARIANT, "stage": STAGE, "loaded": True, "version": _model.version,
            "source": _model.source, "threshold": _model.threshold, "nb_samp": _model.nb_samp}


@app.post("/reload")
def reload():
    m = load_model()
    return {"variant": VARIANT, "loaded": m is not None, "version": m.version if m else None}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, f"No model loaded for variant {VARIANT}")
    t0 = time.perf_counter()
    try:
        wav = trim_silence(decode_bytes(await file.read()))
        res = _model.predict(wav)
    except AudioError as e:
        INFER_TOTAL.labels(VARIANT, "bad_audio").inc()
        raise HTTPException(400, str(e))
    except Exception:
        INFER_TOTAL.labels(VARIANT, "error").inc()
        raise
    dt = time.perf_counter() - t0
    INFER_SECONDS.labels(VARIANT).observe(dt)
    INFER_TOTAL.labels(VARIANT, "ok").inc()
    return {
        "variant": VARIANT, "model_version": _model.version,
        "prob_fake": res["prob_fake"], "margin": res["margin"],
        "segments": res["segments"], "n_windows": res["n_windows"],
        "embedding": [round(float(v), 5) for v in res["embedding"]],
        "inference_ms": round(dt * 1000, 1),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
