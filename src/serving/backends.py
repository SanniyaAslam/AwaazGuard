"""How the gateway reaches the two models (champion = Production, candidate = Staging).

* EmbeddedBackend (default) - both ONNX models live inside the gateway process.
  Smallest footprint: the whole app fits on a free 1 GB VM.
* HttpBackend - each model runs in its own inference_app container (CHAMPION_URL /
  CANDIDATE_URL). Use this when you scale out to bigger machines.

Select with MODEL_BACKEND=embedded|http.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time

import httpx
import numpy as np
from fastapi import HTTPException
from prometheus_client import Gauge, Histogram

from src.serving.audio import to_wav_bytes
from src.serving.model_store import LoadedModel, load_from_dir, load_from_registry
from src.utils.common import get_logger

log = get_logger("backend")
STAGES = {"champion": "Production", "candidate": "Staging"}
INFER_SECONDS = Histogram("awaaz_inference_seconds", "Model inference latency", ["variant"],
                          buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16))
MODEL_VERSION = Gauge("awaaz_model_loaded", "1 if a model is loaded", ["variant", "version"])


class EmbeddedBackend:
    def __init__(self) -> None:
        self.name = os.getenv("MODEL_NAME", "awaazguard-aasist")
        self.slots: dict[str, LoadedModel | None] = {"champion": None, "candidate": None}
        self.lock = threading.Lock()
        self.batch = int(os.getenv("ORT_BATCH", "1"))
        # one clip at a time keeps peak memory ~550 MB (free 1 GB VMs); raise on bigger machines
        self.gate = threading.Semaphore(int(os.getenv("MAX_CONCURRENT_INFERENCE", "1")))

    def _load(self, variant: str) -> LoadedModel | None:
        local = os.getenv(f"MODEL_DIR_{variant.upper()}", "")
        try:
            return load_from_dir(local) if local else load_from_registry(self.name, STAGES[variant])
        except Exception as e:  # registry unreachable -> keep what we have
            log.error("Loading %s failed: %s", variant, e)
            return self.slots[variant]

    def reload(self, variant: str) -> dict:
        m = self._load(variant)
        with self.lock:
            old = self.slots[variant]
            if old is not None:
                MODEL_VERSION.labels(variant, old.version).set(0)
            self.slots[variant] = m
        if m:
            MODEL_VERSION.labels(variant, m.version).set(1)
            log.info("%s loaded: %s", variant, m.source)
        return {"variant": variant, "loaded": m is not None, "version": m.version if m else None}

    def startup(self) -> None:
        for v in self.slots:
            self.reload(v)

    def info(self, variant: str) -> dict:
        m = self.slots[variant]
        if m is None:
            return {"variant": variant, "stage": STAGES[variant], "loaded": False}
        return {"variant": variant, "stage": STAGES[variant], "loaded": True, "version": m.version,
                "source": m.source, "threshold": m.threshold, "nb_samp": m.nb_samp}

    def available(self, variant: str) -> bool:
        return self.slots[variant] is not None

    def model(self, variant: str) -> LoadedModel | None:
        return self.slots[variant]

    async def infer(self, variant: str, wav: np.ndarray) -> dict:
        m = self.slots[variant]
        if m is None:
            raise HTTPException(503, f"No model loaded for {variant}")
        t0 = time.perf_counter()
        def run():
            with self.gate:
                return m.predict(wav, batch=self.batch)

        res = await asyncio.get_running_loop().run_in_executor(None, run)
        dt = time.perf_counter() - t0
        INFER_SECONDS.labels(variant).observe(dt)
        return {"variant": variant, "model_version": m.version, "prob_fake": res["prob_fake"],
                "margin": res["margin"], "segments": res["segments"],
                "embedding": [round(float(v), 5) for v in res["embedding"]],
                "inference_ms": round(dt * 1000, 1)}


class HttpBackend:
    def __init__(self) -> None:
        self.urls = {"champion": os.getenv("CHAMPION_URL", "http://localhost:8001"),
                     "candidate": os.getenv("CANDIDATE_URL", "http://localhost:8002")}
        self.cache: dict[str, dict] = {}

    def startup(self) -> None:
        pass

    def info(self, variant: str) -> dict:
        try:
            self.cache[variant] = httpx.get(f"{self.urls[variant]}/info", timeout=5).json()
        except Exception as e:
            self.cache[variant] = {"loaded": False, "error": type(e).__name__}
        return self.cache[variant]

    def available(self, variant: str) -> bool:
        return True  # failures fall back to the champion in the gateway

    def model(self, variant: str):
        return None

    def reload(self, variant: str) -> dict:
        return httpx.post(f"{self.urls[variant]}/reload", timeout=180).json()

    async def infer(self, variant: str, wav: np.ndarray) -> dict:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{self.urls[variant]}/infer",
                             files={"file": ("clip.wav", to_wav_bytes(wav), "audio/wav")})
        if r.status_code == 400:
            raise HTTPException(400, r.json().get("detail", "bad audio"))
        r.raise_for_status()
        return r.json()


def make_backend():
    return HttpBackend() if os.getenv("MODEL_BACKEND", "embedded") == "http" else EmbeddedBackend()
