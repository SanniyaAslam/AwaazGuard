"""Move a registered model version between stages (manual control of the registry).

  python scripts/promote.py --list
  python scripts/promote.py --version 3 --stage Production   # make v3 the champion
  python scripts/promote.py --version 4 --stage Archived     # retire a candidate
"""
from __future__ import annotations

import argparse

from mlflow.tracking import MlflowClient

from src.utils.common import load_params
from src.utils.tracking import setup_mlflow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=str)
    ap.add_argument("--stage", choices=["Staging", "Production", "Archived", "None"])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    params = load_params()
    setup_mlflow(params["mlflow"]["experiment_name"])
    name = params["register"]["model_name"]
    client = MlflowClient()

    if a.version and a.stage:
        client.transition_model_version_stage(
            name, a.version, a.stage, archive_existing_versions=a.stage == "Production")
        print(f"{name} v{a.version} -> {a.stage}")

    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda m: int(m.version)):
        tags = mv.tags or {}
        print(f"v{mv.version:>3}  {mv.current_stage:<10}  eval_eer={tags.get('eval_eer', '-'):<8} run={mv.run_id[:8]}")


if __name__ == "__main__":
    main()
