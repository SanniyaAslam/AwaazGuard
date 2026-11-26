"""AwaazGuard app server: website, canary routing, prediction log, feedback, admin, metrics.

            ┌──────── weight w ────────► candidate model (registry stage Staging)
  client ──►│ gateway
            └──────── 1 - w ───────────► champion model  (registry stage Production)

By default both models run inside this process (MODEL_BACKEND=embedded), the database is
SQLite, the website is served from FRONTEND_DIR and the drift monitor runs as a background
thread (MONITOR_IN_PROCESS=1). Everything fits on a free 1 GB virtual machine.

Env vars: MODEL_BACKEND, MODEL_NAME, ADMIN_TOKEN, DATABASE_URL, CANARY_WEIGHT, MAX_AUDIO_S (30),
          RISK_SUSPICIOUS (0.35), RISK_FAKE (0.65), DRIFT_REPORT_DIR, FRONTEND_DIR,
          MONITOR_IN_PROCESS, MONITOR_INTERVAL_MIN, S3_BUCKET (optional)
Run locally:  uvicorn src.serving.gateway_app:app --port 8000
"""
from __future__ import annotations

import os
import random
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest)
from prometheus_client.core import REGISTRY, GaugeMetricFamily
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from starlette.responses import Response

from src.serving.audio import AudioError, audio_stats, decode_bytes, to_wav_bytes, trim_silence
from src.serving.backends import make_backend
from src.serving.db import (CanaryEvent, DriftRun, Feedback, Prediction, SessionLocal, init_db)
from src.serving.storage import Storage
from src.utils.common import get_logger
from src.utils.metrics import compute_eer

log = get_logger("gateway")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")
RISK_SUSPICIOUS = float(os.getenv("RISK_SUSPICIOUS", "0.35"))
RISK_FAKE = float(os.getenv("RISK_FAKE", "0.65"))
DRIFT_REPORT_DIR = Path(os.getenv("DRIFT_REPORT_DIR", "./drift_reports"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_AUDIO_S = float(os.getenv("MAX_AUDIO_S", "30"))
FRONTEND_DIR = os.getenv("FRONTEND_DIR", "")

REQS = Counter("awaaz_requests_total", "Predictions served", ["variant", "verdict"])
ERRS = Counter("awaaz_errors_total", "Failed predictions", ["variant", "kind"])
LAT = Histogram("awaaz_request_seconds", "End-to-end latency", ["variant"],
                buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
PFAKE = Histogram("awaaz_prob_fake", "Predicted fake probability", ["variant"],
                  buckets=[i / 10 for i in range(1, 11)])
FEEDBACK = Counter("awaaz_feedback_total", "Feedback received", ["label", "agrees"])
CANARY_W = Gauge("awaaz_canary_weight", "Share of traffic sent to the candidate")


class DriftCollector:
    """Exposes the latest drift run (written by the monitor service) at scrape time."""

    def collect(self):
        g = {k: GaugeMetricFamily(f"awaaz_drift_{k}", f"Latest drift {k}")
             for k in ("mmd", "mmd_p_value", "ood_rate", "features_share", "detected")}
        try:
            with SessionLocal() as s:
                d = s.scalars(select(DriftRun).order_by(DriftRun.id.desc()).limit(1)).first()
            if d:
                g["mmd"].add_metric([], d.mmd)
                g["mmd_p_value"].add_metric([], d.mmd_p_value)
                g["ood_rate"].add_metric([], d.ood_rate)
                g["features_share"].add_metric([], d.drifted_features_share)
                g["detected"].add_metric([], float(d.drift_detected))
        except Exception:
            pass
        yield from g.values()


REGISTRY.register(DriftCollector())

app = FastAPI(title="AwaazGuard Gateway", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
storage = Storage()
backend = make_backend()
state = {"canary_weight": float(os.getenv("CANARY_WEIGHT", "0"))}
CANARY_W.set(state["canary_weight"])


def admin(x_admin_token: str = Header(default=""), token: str = ""):
    """Admin auth via `X-Admin-Token` header (or ?token= for links opened in a browser tab)."""
    if ADMIN_TOKEN and (x_admin_token or token) != ADMIN_TOKEN:
        raise HTTPException(401, "Invalid admin token")


def _monitor_thread():
    """Run each drift check in a short-lived subprocess: Evidently/pandas memory is returned
    to the OS afterwards, which keeps the app inside a free 1 GB VM."""
    import subprocess
    import sys

    interval = float(os.getenv("MONITOR_INTERVAL_MIN", "15")) * 60
    while True:
        time.sleep(interval)
        p = subprocess.run([sys.executable, "-m", "src.monitoring.monitor_loop", "--once"],
                           capture_output=True, text=True, timeout=600)
        out = (p.stdout + p.stderr).strip().splitlines()
        log.info("Drift check finished (exit %s): %s", p.returncode, out[-1] if out else "")


@app.on_event("startup")
def _startup():
    init_db()
    backend.startup()
    if os.getenv("MONITOR_IN_PROCESS", "1") == "1":
        threading.Thread(target=_monitor_thread, daemon=True, name="drift-monitor").start()
    with SessionLocal() as s:  # restore last canary weight after restarts
        ev = s.scalars(select(CanaryEvent).order_by(CanaryEvent.id.desc()).limit(1)).first()
        if ev:
            state["canary_weight"] = ev.weight
            CANARY_W.set(ev.weight)


def verdict_for(p: float) -> str:
    if p >= RISK_FAKE:
        return "likely_fake"
    if p >= RISK_SUSPICIOUS:
        return "suspicious"
    return "safe"


# ----------------------------------------------------------------- public API
@app.get("/api/health")
def health():
    return {"gateway": "ok", "canary_weight": state["canary_weight"],
            "champion": backend.info("champion"), "candidate": backend.info("candidate")}


@app.get("/api/config")
def public_config():
    """Links shown on the Operations page (set at runtime, so one image fits every server)."""
    mlflow_url = os.getenv("PUBLIC_MLFLOW_URL") or os.getenv("MLFLOW_TRACKING_URI", "")
    return {"grafana_url": os.getenv("PUBLIC_GRAFANA_URL", ""),
            "mlflow_url": mlflow_url if mlflow_url.startswith("http") else "",
            "max_audio_s": MAX_AUDIO_S}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1e6:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    try:
        wav = trim_silence(decode_bytes(raw, max_seconds=MAX_AUDIO_S))
    except AudioError as e:
        ERRS.labels("none", "bad_audio").inc()
        raise HTTPException(400, str(e))
    wav_bytes = to_wav_bytes(wav)

    use_candidate = random.random() < state["canary_weight"] and backend.available("candidate")
    variant = "candidate" if use_candidate else "champion"
    t0 = time.perf_counter()
    try:
        res = await backend.infer(variant, wav)
    except HTTPException as e:
        if e.status_code == 400 or not use_candidate:
            ERRS.labels(variant, "unavailable").inc()
            raise
        variant, res = "champion", await backend.infer("champion", wav)
    except Exception as e:
        ERRS.labels(variant, "model_error").inc()
        if not use_candidate:
            raise HTTPException(503, "Model unavailable")
        log.warning("Candidate failed (%s) - falling back to champion", e)
        variant, res = "champion", await backend.infer("champion", wav)
    latency = time.perf_counter() - t0

    verdict = verdict_for(res["prob_fake"])
    key = f"{time.strftime('%Y/%m/%d')}/{uuid.uuid4().hex}.wav"
    audio_uri = storage.put(key, wav_bytes)
    stats = audio_stats(wav)
    with SessionLocal() as s:
        row = Prediction(variant=variant, model_version=res["model_version"],
                         prob_fake=res["prob_fake"], margin=res["margin"], verdict=verdict,
                         duration_s=stats["duration_s"], rms=stats["rms"],
                         clipped_fraction=stats["clipped_fraction"], latency_ms=latency * 1000,
                         filename=(file.filename or "")[:255], audio_key=audio_uri,
                         segments=res["segments"], embedding=res["embedding"])
        s.add(row)
        s.commit()
        pid = row.id

    REQS.labels(variant, verdict).inc()
    LAT.labels(variant).observe(latency)
    PFAKE.labels(variant).observe(res["prob_fake"])
    return {"id": pid, "verdict": verdict, "prob_fake": round(res["prob_fake"], 4),
            "segments": res["segments"], "duration_s": stats["duration_s"],
            "model_version": res["model_version"], "variant": variant,
            "latency_ms": round(latency * 1000, 1)}


class FeedbackIn(BaseModel):
    prediction_id: int
    label: str = Field(pattern="^(bonafide|spoof)$")
    source: str = "user"


@app.post("/api/feedback")
def feedback(fb: FeedbackIn):
    with SessionLocal() as s:
        pred = s.get(Prediction, fb.prediction_id)
        if not pred:
            raise HTTPException(404, "Unknown prediction id")
        s.add(Feedback(prediction_id=fb.prediction_id, label=fb.label, source=fb.source[:32]))
        s.commit()
    predicted_fake = pred.prob_fake >= 0.5
    FEEDBACK.labels(fb.label, str(predicted_fake == (fb.label == "spoof")).lower()).inc()
    return {"ok": True}


# ----------------------------------------------------------------- admin API
@app.get("/api/admin/predictions", dependencies=[Depends(admin)])
def recent(limit: int = 50):
    with SessionLocal() as s:
        rows = s.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(min(limit, 500))).all()
        fb = {f.prediction_id: f.label for f in s.scalars(
            select(Feedback).where(Feedback.prediction_id.in_([r.id for r in rows])))}
    return [{"id": r.id, "created_at": r.created_at.isoformat(), "variant": r.variant,
             "model_version": r.model_version, "verdict": r.verdict,
             "prob_fake": round(r.prob_fake, 4), "duration_s": r.duration_s,
             "latency_ms": round(r.latency_ms, 1), "filename": r.filename,
             "feedback": fb.get(r.id)} for r in rows]


def variant_stats(s, variant: str, since_id: int = 0) -> dict:
    q = select(Prediction).where(Prediction.variant == variant, Prediction.id > since_id)
    preds = s.scalars(q).all()
    if not preds:
        return {"variant": variant, "requests": 0}
    lat = np.array([p.latency_ms for p in preds])
    labelled = s.execute(select(Prediction.margin, Feedback.label).join(
        Feedback, Feedback.prediction_id == Prediction.id).where(
        Prediction.variant == variant, Prediction.id > since_id)).all()
    bona = np.array([m for m, lab in labelled if lab == "bonafide"])
    spoof = np.array([m for m, lab in labelled if lab == "spoof"])
    eer = compute_eer(bona, spoof)[0] if len(bona) and len(spoof) else None
    return {"variant": variant, "requests": len(preds),
            "model_versions": sorted({p.model_version for p in preds}),
            "latency_p50_ms": float(np.percentile(lat, 50)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "fake_rate": float(np.mean([p.prob_fake >= 0.5 for p in preds])),
            "labelled": len(labelled), "eer_on_feedback": eer}


@app.get("/api/admin/stats", dependencies=[Depends(admin)])
def stats(since_id: int = 0):
    with SessionLocal() as s:
        total = s.scalar(select(func.count(Prediction.id))) or 0
        n_fb = s.scalar(select(func.count(Feedback.id))) or 0
        last_drift = s.scalars(select(DriftRun).order_by(DriftRun.id.desc()).limit(1)).first()
        events = s.scalars(select(CanaryEvent).order_by(CanaryEvent.id.desc()).limit(10)).all()
        out = {
            "total_predictions": total, "total_feedback": n_fb,
            "canary_weight": state["canary_weight"],
            "champion": variant_stats(s, "champion", since_id),
            "candidate": variant_stats(s, "candidate", since_id),
            "drift": None if not last_drift else {
                "created_at": last_drift.created_at.isoformat(), "mmd": last_drift.mmd,
                "mmd_p_value": last_drift.mmd_p_value, "ood_rate": last_drift.ood_rate,
                "drifted_features_share": last_drift.drifted_features_share,
                "drift_detected": last_drift.drift_detected, "n_current": last_drift.n_current},
            "canary_events": [{"at": e.created_at.isoformat(), "action": e.action,
                               "weight": e.weight, "detail": e.detail} for e in events],
        }
    return out


class CanaryIn(BaseModel):
    weight: float = Field(ge=0.0, le=1.0)
    action: str = "weight"
    detail: str = ""


@app.post("/api/admin/canary", dependencies=[Depends(admin)])
def set_canary(c: CanaryIn):
    state["canary_weight"] = c.weight
    CANARY_W.set(c.weight)
    with SessionLocal() as s:
        s.add(CanaryEvent(action=c.action[:32], weight=c.weight, detail=c.detail))
        s.commit()
    log.info("Canary %s -> weight %.2f (%s)", c.action, c.weight, c.detail)
    return {"canary_weight": c.weight}


@app.post("/api/admin/reload/{variant}", dependencies=[Depends(admin)])
async def reload(variant: str):
    if variant not in ("champion", "candidate"):
        raise HTTPException(404, "variant must be champion or candidate")
    import asyncio

    return await asyncio.get_running_loop().run_in_executor(None, backend.reload, variant)


@app.get("/api/admin/feedback-export", dependencies=[Depends(admin)])
def feedback_export(only_new: bool = True):
    """Labelled production clips for retraining (latest label per prediction wins)."""
    with SessionLocal() as s:
        q = select(Feedback, Prediction).join(Prediction, Feedback.prediction_id == Prediction.id)
        if only_new:
            q = q.where(Feedback.exported.is_(False))
        rows = s.execute(q.order_by(Feedback.id)).all()
    latest = {}
    for f, p in rows:
        latest[p.id] = {"feedback_id": f.id, "prediction_id": p.id, "label": f.label,
                        "source": f.source, "audio_uri": p.audio_key}
    return list(latest.values())


@app.post("/api/admin/feedback-export/ack", dependencies=[Depends(admin)])
def feedback_ack(ids: list[int]):
    with SessionLocal() as s:
        for f in s.scalars(select(Feedback).where(Feedback.id.in_(ids))):
            f.exported = True
        s.commit()
    return {"acknowledged": len(ids)}


@app.get("/api/admin/audio/{prediction_id}", dependencies=[Depends(admin)])
def get_audio(prediction_id: int):
    with SessionLocal() as s:
        p = s.get(Prediction, prediction_id)
    if not p:
        raise HTTPException(404)
    return Response(storage.get(p.audio_key), media_type="audio/wav")


@app.get("/api/admin/drift-report", dependencies=[Depends(admin)])
def drift_report():
    path = DRIFT_REPORT_DIR / "latest.html"
    if not path.exists():
        return HTMLResponse("<p>No drift report yet.</p>")
    return FileResponse(path, media_type="text/html")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ----------------------------------------------------------------- website (static Next.js export)
if FRONTEND_DIR and Path(FRONTEND_DIR).is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
