"""Tests for serving, gateway, drift, red-team channels and canary decisions.

A tiny random-weight AASIST-L is exported to ONNX once per session, so no real model,
dataset, MLflow server or database is needed (SQLite in a temp folder).
"""
import importlib
import os

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from src.serving.audio import decode_bytes, make_windows, to_wav_bytes


def _tone(sec=2.5, sr=16000, f=220):
    t = np.arange(int(sec * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    from src.models.aasist import Model

    cfg = yaml.safe_load(open("params.yaml"))["main"]["model_config"]
    torch.manual_seed(0)
    m = Model(cfg).eval()
    d = tmp_path_factory.mktemp("serving")
    torch.onnx.export(m, (torch.randn(2, cfg["nb_samp"]),), str(d / "model.onnx"),
                      input_names=["wave"], output_names=["embedding", "logits"], opset_version=17,
                      dynamic_axes={"wave": {0: "b"}, "embedding": {0: "b"}, "logits": {0: "b"}},
                      dynamo=False)
    rng = np.random.default_rng(0)
    np.savez(d / "reference.npz", embeddings=rng.normal(size=(60, 160)).astype(np.float32),
             margins=rng.normal(size=60).astype(np.float32), labels=np.zeros(60, np.int8),
             threshold=np.float32(0.0), nb_samp=np.int32(cfg["nb_samp"]))
    return d


def test_decode_and_windows():
    wav = decode_bytes(to_wav_bytes(_tone(5.2)))
    assert abs(len(wav) / 16000 - 5.2) < 0.01
    w, starts = make_windows(wav, 64600, 16000)
    assert w.shape[1] == 64600 and starts[-1] + 64600 == len(wav)
    w2, _ = make_windows(wav[:8000], 64600, 16000)
    assert w2.shape == (1, 64600)


def test_model_predict_segments(model_dir):
    from src.serving.model_store import load_from_dir

    m = load_from_dir(model_dir)
    r = m.predict(_tone(6.3))
    assert 0 <= r["prob_fake"] <= 1
    assert len(r["segments"]) == 7 and r["embedding"].shape == (160,)


@pytest.fixture()
def gateway(model_dir, tmp_path, monkeypatch):
    """Gateway in the default free-tier mode: both models embedded, SQLite, website mounted."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>AwaazGuard</h1>")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("AUDIO_STORE_DIR", str(tmp_path / "audio"))
    monkeypatch.setenv("ADMIN_TOKEN", "t0k")
    monkeypatch.setenv("MODEL_DIR_CHAMPION", str(model_dir))
    monkeypatch.setenv("MODEL_DIR_CANDIDATE", str(model_dir))
    monkeypatch.setenv("MONITOR_IN_PROCESS", "0")
    monkeypatch.setenv("FRONTEND_DIR", str(site))
    from prometheus_client import REGISTRY

    import src.serving.backends as be
    import src.serving.db as db
    import src.serving.gateway_app as gw
    import src.serving.storage as st

    for c in list(REGISTRY._collector_to_names):     # avoid duplicate metric registration on reload
        REGISTRY.unregister(c)
    for mod in (db, st, be, gw):                     # re-read the environment set above
        importlib.reload(mod)
    with TestClient(gw.app) as c:
        yield c


def test_gateway_predict_feedback_admin(gateway):
    r = gateway.post("/api/predict", files={"file": ("x.wav", to_wav_bytes(_tone(3)))})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] in {"safe", "suspicious", "likely_fake"} and body["segments"]
    assert gateway.post("/api/feedback", json={"prediction_id": body["id"], "label": "spoof"}).json()["ok"]
    assert gateway.post("/api/feedback", json={"prediction_id": 999, "label": "spoof"}).status_code == 404
    assert gateway.get("/api/admin/stats").status_code == 401
    s = gateway.get("/api/admin/stats", headers={"X-Admin-Token": "t0k"}).json()
    assert s["total_predictions"] == 1 and s["total_feedback"] == 1
    exp = gateway.get("/api/admin/feedback-export", headers={"X-Admin-Token": "t0k"}).json()
    assert exp[0]["label"] == "spoof"
    audio = gateway.get(f"/api/admin/audio/{body['id']}", params={"token": "t0k"})
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert gateway.post("/api/predict", files={"file": ("x.wav", b"not audio")}).status_code == 400


def test_website_and_health(gateway):
    assert "AwaazGuard" in gateway.get("/").text
    h = gateway.get("/api/health").json()
    assert h["champion"]["loaded"] and h["candidate"]["loaded"]
    assert gateway.get("/api/config").json()["max_audio_s"] == 30


def test_canary_routes_to_candidate(gateway):
    h = {"X-Admin-Token": "t0k"}
    gateway.post("/api/admin/canary", json={"weight": 1.0, "action": "start"}, headers=h)
    r = gateway.post("/api/predict", files={"file": ("x.wav", to_wav_bytes(_tone(2)))}).json()
    assert r["variant"] == "candidate"
    assert gateway.post("/api/admin/reload/candidate", headers=h).json()["loaded"]


def test_canary_weight_and_metrics(gateway):
    h = {"X-Admin-Token": "t0k"}
    assert gateway.post("/api/admin/canary", json={"weight": 0.25, "action": "start"}, headers=h).json()["canary_weight"] == 0.25
    assert gateway.post("/api/admin/canary", json={"weight": 2}, headers=h).status_code == 422
    m = gateway.get("/metrics").text
    assert "awaaz_canary_weight 0.25" in m


def test_drift_detects_shift(tmp_path):
    from src.monitoring.drift import mmd_test, ood_rate, run_drift

    rng = np.random.default_rng(1)
    ref = rng.normal(size=(150, 16))
    same = rng.normal(size=(80, 16))
    shifted = rng.normal(loc=1.5, size=(80, 16))
    assert mmd_test(ref, same, n_perm=100)[1] > 0.05
    assert mmd_test(ref, shifted, n_perm=100)[1] < 0.05
    assert ood_rate(ref, shifted) > ood_rate(ref, same)
    res = run_drift(ref.astype(np.float32), rng.normal(size=150), shifted.astype(np.float32),
                    rng.normal(size=80) + 2, tmp_path)
    assert res["drift_detected"] and (tmp_path / "latest.html").exists()


def test_channels_keep_length_roughly():
    from src.redteam import channel

    rng = np.random.default_rng(0)
    wav = _tone(2.0)
    for kind in channel.CHANNELS:
        out = channel.apply(wav, kind, rng)
        assert out.dtype == np.float32 and abs(len(out) - len(wav)) < 1600 and np.abs(out).max() <= 1.0


def test_canary_decisions():
    from flows.canary_flow import decide

    base = {"canary_weight": 0.1,
            "champion": {"requests": 500, "latency_p95_ms": 400, "labelled": 50, "eer_on_feedback": 0.10},
            "candidate": {"requests": 60, "latency_p95_ms": 420, "labelled": 30, "eer_on_feedback": 0.08}}
    assert decide.fn(base, 50, 20, 1.5, 0.02)[0] == "promote"
    slow = {**base, "candidate": {**base["candidate"], "latency_p95_ms": 900}}
    assert decide.fn(slow, 50, 20, 1.5, 0.02)[0] == "rollback"
    worse = {**base, "candidate": {**base["candidate"], "eer_on_feedback": 0.2}}
    assert decide.fn(worse, 50, 20, 1.5, 0.02)[0] == "rollback"
    few = {**base, "candidate": {**base["candidate"], "requests": 3}}
    assert decide.fn(few, 50, 20, 1.5, 0.02)[0] == "wait"
    assert decide.fn({**base, "canary_weight": 0}, 50, 20, 1.5, 0.02)[0] == "idle"
