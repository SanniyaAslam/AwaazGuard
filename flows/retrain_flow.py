"""Prefect flow: drift check -> export feedback -> red-team batch -> dvc repro (retrain +
register candidate in Staging) -> red-team gate -> start canary.

Run where you train (GPU machine / Kaggle) with .env loaded:
  python -m flows.retrain_flow --gateway http://<server-address> --token <ADMIN_TOKEN>
  python -m flows.retrain_flow ... --force            # retrain even without drift
  python -m flows.retrain_flow ... --serve            # run every Monday 03:00 (Prefect schedule)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import httpx
from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd: str) -> str:
    logger = get_run_logger()
    logger.info("$ %s", cmd)
    p = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    if p.stdout.strip():
        logger.info(p.stdout[-3000:])
    if p.returncode != 0:
        logger.error(p.stderr[-3000:])
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p.stdout


@task
def check_drift(gateway: str, token: str) -> dict:
    s = httpx.get(f"{gateway}/api/admin/stats", headers={"X-Admin-Token": token}, timeout=30).json()
    return s.get("drift") or {}


@task
def export_feedback(gateway: str, token: str) -> int:
    from scripts.export_feedback import export

    return export(gateway, token, ROOT / "data" / "feedback")


@task
def make_redteam_batch(generators: str, n_fake: int, n_real: int) -> None:
    sh(f"python -m src.redteam.run_redteam generate --generators {generators} "
       f"--n-fake {n_fake} --n-real {n_real}")


@task
def version_data() -> None:
    sh("dvc add data/feedback data/redteam")


@task(retries=0)
def retrain() -> dict:
    sh("dvc repro")
    return json.loads((ROOT / "reports" / "registry.json").read_text())


@task
def redteam_gate() -> bool:
    p = subprocess.run("python -m src.redteam.run_redteam gate --archive-on-fail", shell=True, cwd=ROOT)
    return p.returncode == 0


@task
def start_canary(gateway: str, token: str, weight: float, version: str) -> None:
    h = {"X-Admin-Token": token}
    with httpx.Client(base_url=gateway, headers=h, timeout=180) as c:
        info = c.post("/api/admin/reload/candidate").json()
        if not info.get("loaded"):
            raise RuntimeError(f"Candidate did not load: {info}")
        since = c.get("/api/admin/stats").json()["total_predictions"]
        c.post("/api/admin/canary", json={"weight": weight, "action": "start",
                                          "detail": json.dumps({"version": version, "since_id": since})})


@task
def publish(commit: bool) -> None:
    """Upload new data/model versions. Failures are reported but do not undo the release."""
    log = get_run_logger()
    try:
        sh("dvc push")
        if commit:
            sh("git add dvc.lock reports data/feedback.dvc data/redteam.dvc params.yaml && "
               "git commit -m 'Retrain: new feedback + red-team batch' && git push")
    except RuntimeError as e:
        log.warning("Publishing failed (%s) - run `dvc push` / git push manually", e)


@flow(name="awaazguard-retrain")
def retrain_flow(gateway: str, token: str, force: bool = False, generators: str = "piper",
                 n_fake: int = 200, n_real: int = 200, canary_weight: float = 0.1,
                 commit: bool = False) -> str:
    log = get_run_logger()
    drift = check_drift(gateway, token)
    if not force and not drift.get("drift_detected"):
        log.info("No drift detected (latest: %s) - nothing to do", drift)
        return "skipped"
    n_new = export_feedback(gateway, token)
    log.info("%d new labelled production clips", n_new)
    make_redteam_batch(generators, n_fake, n_real)
    version_data()
    reg = retrain()
    if not redteam_gate():
        log.warning("Candidate v%s blocked by the red-team gate", reg["version"])
        publish(commit)
        return "blocked"
    start_canary(gateway, token, canary_weight, str(reg["version"]))
    publish(commit)
    log.info("Canary started for v%s at %.0f%% traffic", reg["version"], canary_weight * 100)
    return "canary_started"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default=os.getenv("GATEWAY_URL", "http://localhost"))
    ap.add_argument("--token", default=os.getenv("ADMIN_TOKEN", ""))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--generators", default="piper")
    ap.add_argument("--n-fake", type=int, default=200)
    ap.add_argument("--n-real", type=int, default=200)
    ap.add_argument("--canary-weight", type=float, default=0.1)
    ap.add_argument("--commit", action="store_true", help="git commit + push dvc files after")
    ap.add_argument("--serve", action="store_true", help="run on a weekly schedule")
    a = ap.parse_args()
    kw = dict(gateway=a.gateway, token=a.token, force=a.force, generators=a.generators,
              n_fake=a.n_fake, n_real=a.n_real, canary_weight=a.canary_weight, commit=a.commit)
    if a.serve:
        retrain_flow.serve(name="weekly-retrain", cron="0 3 * * 1", parameters=kw)
    else:
        print(retrain_flow(**kw))


if __name__ == "__main__":
    main()
