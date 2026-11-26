"""Model quality gate used in CI: score N eval clips (unseen attacks) with a registry model.

  dvc pull data/processed
  python -m scripts.eval_registry_model --stage Production --n 200 --max-eer 0.35
Exit code 1 if the EER is above --max-eer.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import soundfile as sf

from src.serving.model_store import load_from_registry
from src.utils.common import METADATA_CSV, ROOT, load_params
from src.utils.metrics import compute_eer
from src.utils.tracking import setup_mlflow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="Production")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-eer", type=float, default=0.35)
    a = ap.parse_args()
    params = load_params()
    setup_mlflow(params["mlflow"]["experiment_name"])
    model = load_from_registry(params["register"]["model_name"], a.stage)
    if model is None:
        sys.exit(f"No model in stage {a.stage}")
    meta = pd.read_csv(METADATA_CSV)
    ev = meta[meta["split"] == "eval"].sample(n=min(a.n, (meta["split"] == "eval").sum()), random_state=0)
    margins = [model.predict(sf.read(ROOT / p, dtype="float32")[0])["margin"] for p in ev["path"]]
    ev = ev.assign(margin=margins)
    eer = compute_eer(ev.loc[ev.label == "bonafide", "margin"], ev.loc[ev.label == "spoof", "margin"])[0]
    print(f"{a.stage} v{model.version}: EER on {len(ev)} eval clips = {eer:.4f} (limit {a.max_eer})")
    sys.exit(1 if eer > a.max_eer else 0)


if __name__ == "__main__":
    main()
