"""Prefect flow: decide whether the canary candidate is promoted or rolled back.

Rules (all measured only on traffic since the canary started):
  * wait until the candidate has >= min_requests predictions
  * ROLLBACK if candidate p95 latency > max_latency_ratio x champion p95
  * ROLLBACK if candidate EER on user feedback > champion EER + max_eer_increase
    (needs >= min_labelled labelled predictions per variant, else latency-only decision)
  * otherwise PROMOTE: Staging -> Production in MLflow, reload champion, canary weight -> 0

  python -m flows.canary_flow --gateway http://<server-address> --token <ADMIN_TOKEN>
  python -m flows.canary_flow ... --serve    # check every 30 minutes
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx
from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parents[1]


@task
def get_stats(gateway: str, token: str) -> dict:
    h = {"X-Admin-Token": token}
    first = httpx.get(f"{gateway}/api/admin/stats", headers=h, timeout=30).json()
    start = next((e for e in first["canary_events"] if e["action"] == "start"), None)
    since = json.loads(start["detail"]).get("since_id", 0) if start else 0
    s = httpx.get(f"{gateway}/api/admin/stats", headers=h, params={"since_id": since}, timeout=30).json()
    s["_start_event"] = start
    return s


@task
def decide(s: dict, min_requests: int, min_labelled: int, max_latency_ratio: float,
           max_eer_increase: float) -> tuple[str, str]:
    if s["canary_weight"] <= 0:
        return "idle", "no canary running"
    c, h = s["candidate"], s["champion"]
    if c.get("requests", 0) < min_requests:
        return "wait", f"candidate has {c.get('requests', 0)}/{min_requests} requests"
    if h.get("requests") and c["latency_p95_ms"] > max_latency_ratio * h["latency_p95_ms"]:
        return "rollback", f"p95 latency {c['latency_p95_ms']:.0f} ms vs {h['latency_p95_ms']:.0f} ms"
    ce, he = c.get("eer_on_feedback"), h.get("eer_on_feedback")
    if c.get("labelled", 0) >= min_labelled and h.get("labelled", 0) >= min_labelled \
            and ce is not None and he is not None:
        if ce > he + max_eer_increase:
            return "rollback", f"feedback EER {ce:.3f} vs champion {he:.3f}"
        return "promote", f"feedback EER {ce:.3f} vs champion {he:.3f}; latency ok"
    return "promote", "latency ok; not enough labelled feedback for an EER comparison"


@task
def act(action: str, reason: str, gateway: str, token: str) -> None:
    from mlflow.tracking import MlflowClient

    from src.utils.common import load_params
    from src.utils.tracking import setup_mlflow

    params = load_params()
    setup_mlflow(params["mlflow"]["experiment_name"])
    name = params["register"]["model_name"]
    client = MlflowClient()
    cand = client.get_latest_versions(name, stages=["Staging"])
    h = {"X-Admin-Token": token}
    with httpx.Client(base_url=gateway, headers=h, timeout=180) as c:
        if action == "promote" and cand:
            client.transition_model_version_stage(name, cand[0].version, "Production",
                                                  archive_existing_versions=True)
            c.post("/api/admin/canary", json={"weight": 0.0, "action": "promote",
                                              "detail": f"v{cand[0].version}: {reason}"})
            c.post("/api/admin/reload/champion")
            c.post("/api/admin/reload/candidate")
        elif action == "rollback":
            c.post("/api/admin/canary", json={"weight": 0.0, "action": "rollback", "detail": reason})
            if cand:
                client.transition_model_version_stage(name, cand[0].version, "Archived")
            c.post("/api/admin/reload/candidate")


@flow(name="awaazguard-canary")
def canary_flow(gateway: str, token: str, min_requests: int = 50, min_labelled: int = 20,
                max_latency_ratio: float = 1.5, max_eer_increase: float = 0.02) -> str:
    log = get_run_logger()
    s = get_stats(gateway, token)
    action, reason = decide(s, min_requests, min_labelled, max_latency_ratio, max_eer_increase)
    log.info("Canary decision: %s (%s)", action.upper(), reason)
    if action in ("promote", "rollback"):
        act(action, reason, gateway, token)
    return action


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default=os.getenv("GATEWAY_URL", "http://localhost"))
    ap.add_argument("--token", default=os.getenv("ADMIN_TOKEN", ""))
    ap.add_argument("--min-requests", type=int, default=50)
    ap.add_argument("--min-labelled", type=int, default=20)
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    kw = dict(gateway=a.gateway, token=a.token, min_requests=a.min_requests,
              min_labelled=a.min_labelled)
    if a.serve:
        canary_flow.serve(name="canary-check", interval=1800, parameters=kw)
    else:
        print(canary_flow(**kw))


if __name__ == "__main__":
    main()
