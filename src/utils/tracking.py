"""MLflow setup. Requires a REMOTE tracking server (DagsHub) unless explicitly overridden."""
from __future__ import annotations

import os
import subprocess

import mlflow
import yaml
from dotenv import load_dotenv

from src.utils.common import ROOT, get_logger

log = get_logger("tracking")


def setup_mlflow(experiment_name: str) -> str:
    load_dotenv(ROOT / ".env")
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        if os.getenv("AWAAZ_LOCAL_MLFLOW") == "1":
            uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
            log.warning("Using LOCAL MLflow (%s) - smoke-test mode only.", uri)
        else:
            raise RuntimeError(
                "MLFLOW_TRACKING_URI is not set. Copy .env.example to .env and add your "
                "DagsHub MLflow URI, username and access token."
            )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    log.info("MLflow tracking URI: %s | experiment: %s", uri, experiment_name)
    return uri


def lineage_tags() -> dict[str, str]:
    """Git commit + DVC data hash so every run is traceable to exact code and data."""
    tags = {"project": "AwaazGuard", "deliverable": "1"}
    try:
        tags["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        tags["git_commit"] = "unknown"
    lock = ROOT / "dvc.lock"
    if lock.exists():
        try:
            data = yaml.safe_load(lock.read_text())
            for out in data["stages"]["prepare"]["outs"]:
                if out["path"].endswith("data/processed"):
                    tags["data_version_md5"] = out["md5"]
        except Exception:
            pass
    return tags
