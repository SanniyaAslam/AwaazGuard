"""Drift monitor: every MONITOR_INTERVAL_MIN minutes, compare recent champion traffic with
the champion model's reference set and store the result (DB -> Prometheus/Grafana + dashboard).

Env: DATABASE_URL, MODEL_NAME, MODEL_DIR (optional), MONITOR_INTERVAL_MIN (15),
     MONITOR_WINDOW (300 most recent predictions), MONITOR_MIN_SAMPLES (50), DRIFT_REPORT_DIR
In the default deployment this runs as a background thread inside the gateway
(MONITOR_IN_PROCESS=1). It can also run on its own:
Run once:  python -m src.monitoring.monitor_loop --once
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from sqlalchemy import select

from src.monitoring.drift import run_drift
from src.serving.db import DriftRun, Prediction, SessionLocal, init_db
from src.serving.model_store import load_from_dir, load_from_registry
from src.utils.common import get_logger

log = get_logger("monitor")
REPORT_DIR = Path(os.getenv("DRIFT_REPORT_DIR", "./drift_reports"))


def check_once(model=None) -> dict | None:
    """Run one drift check. `model` = the champion already loaded in the gateway (in-process
    mode); when None, the Production model is loaded from MODEL_DIR or the registry."""
    if model is None:
        model_dir = os.getenv("MODEL_DIR", "") or os.getenv("MODEL_DIR_CHAMPION", "")
        model = (load_from_dir(model_dir) if model_dir
                 else load_from_registry(os.getenv("MODEL_NAME", "awaazguard-aasist"), "Production"))
    if model is None:
        log.warning("No Production model - skipping drift check")
        return None
    window = int(os.getenv("MONITOR_WINDOW", "300"))
    with SessionLocal() as s:
        rows = s.scalars(select(Prediction).where(
            Prediction.variant == "champion", Prediction.model_version == model.version)
            .order_by(Prediction.id.desc()).limit(window)).all()
    if len(rows) < max(10, int(os.getenv("MONITOR_MIN_SAMPLES", "50"))):   # statistics need data
        log.info("Only %d predictions for v%s - waiting for more traffic", len(rows), model.version)
        return None

    cur_emb = np.array([r.embedding for r in rows], dtype=np.float32)
    cur_margin = np.array([r.margin for r in rows], dtype=np.float32)
    res = run_drift(model.ref_embeddings, model.ref_margins, cur_emb, cur_margin, REPORT_DIR)

    with SessionLocal() as s:
        s.add(DriftRun(model_version=model.version, **res))
        s.commit()
    level = log.warning if res["drift_detected"] else log.info
    level("Drift check v%s: MMD p=%.4f, OOD rate=%.2f, drifted features=%.2f -> %s",
          model.version, res["mmd_p_value"], res["ood_rate"], res["drifted_features_share"],
          "DRIFT" if res["drift_detected"] else "ok")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    init_db()
    interval = float(os.getenv("MONITOR_INTERVAL_MIN", "15")) * 60
    while True:
        try:
            check_once()
        except Exception as e:
            log.exception("Drift check failed: %s", e)
        if a.once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
