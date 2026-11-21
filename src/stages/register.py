"""Stage 8 - register: push the best AASIST-L checkpoint to the MLflow Model Registry
and move it to the Staging stage (also sets a `staging` alias for MLflow >= 2.9)."""
from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.common import (MODELS_DIR, REPORTS_DIR, get_logger, load_json, load_params,
                              save_json)
from src.utils.tracking import setup_mlflow

log = get_logger("register")


def main() -> None:
    params = load_params()
    rp = params["register"]
    setup_mlflow(params["mlflow"]["experiment_name"])
    client = MlflowClient()

    m_meta = load_json(MODELS_DIR / "main" / "meta.json")
    metrics = load_json(REPORTS_DIR / "metrics.json")["main_aasist_l"]
    run_id = m_meta["run_id"]

    mv = mlflow.register_model(f"runs:/{run_id}/model", rp["model_name"])
    client.update_model_version(
        rp["model_name"], mv.version,
        description=(f"AASIST-L anti-spoofing model. Eval EER on unseen attacks = "
                     f"{metrics['eer']:.4f}, AUC = {metrics['auc']:.4f}. Source run {run_id}."))
    for k, v in {"eval_eer": f"{metrics['eer']:.4f}", "eval_auc": f"{metrics['auc']:.4f}",
                 "dataset": "ASVspoof2019-LA-subset"}.items():
        client.set_model_version_tag(rp["model_name"], mv.version, k, v)

    stage = rp["stage"]
    try:
        client.transition_model_version_stage(rp["model_name"], mv.version, stage=stage,
                                              archive_existing_versions=False)
    except Exception as e:  # stage API removed on some servers -> alias still works
        log.warning("Stage transition not supported (%s); using alias only.", e)
    try:
        client.set_registered_model_alias(rp["model_name"], stage.lower(), mv.version)
    except Exception as e:
        log.warning("Alias not supported by server (%s).", e)

    save_json({"model_name": rp["model_name"], "version": mv.version, "stage": stage,
               "run_id": run_id, "eval_eer": metrics["eer"]}, REPORTS_DIR / "registry.json")
    log.info("Registered %s v%s -> %s", rp["model_name"], mv.version, stage)


if __name__ == "__main__":
    main()
