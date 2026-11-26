"""CI model evaluation gate: champion (Production) vs challenger (Staging) on a FIXED test set.

  dvc pull data/processed
  python -m scripts.model_gate

The test set is a seeded sample of the versioned eval split (unseen attacks), so every pull
request is judged on exactly the same clips. Both models are scored through the real serving
code path (registry download -> ONNX Runtime -> windows -> margin), so a pull request that
breaks audio decoding, windowing or model loading also fails here.

Exit code 1 (deployment blocked) when
  * the Production model is above the absolute EER ceiling, or
  * a Staging challenger is worse than Production by more than the tolerance.
Settings live in params.yaml -> gate and can be overridden on the command line.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import soundfile as sf

from src.utils.common import METADATA_CSV, REPORTS_DIR, ROOT, get_logger, load_params, save_json
from src.utils.gates import decide_gate
from src.utils.metrics import compute_eer

log = get_logger("model_gate")


def fixed_test_set(n: int, seed: int) -> pd.DataFrame:
    meta = pd.read_csv(METADATA_CSV)
    ev = meta[meta["split"] == "eval"]
    return ev.sample(n=min(n, len(ev)), random_state=seed).reset_index(drop=True)


def eer_of(model, waves: list, labels: pd.Series) -> float:
    margins = pd.Series([model.predict(w)["margin"] for w in waves])
    return float(compute_eer(margins[(labels == "bonafide").values], margins[(labels == "spoof").values])[0])


def main() -> None:
    params = load_params()
    g = params.get("gate", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion-stage", default="Production")
    ap.add_argument("--candidate-stage", default="Staging")
    ap.add_argument("--n", type=int, default=g.get("n_clips", 300))
    ap.add_argument("--seed", type=int, default=g.get("seed", 0))
    ap.add_argument("--tolerance", type=float, default=g.get("tolerance", 0.01))
    ap.add_argument("--max-eer", type=float, default=g.get("max_eer", 0.35))
    a = ap.parse_args()

    from src.serving.model_store import load_from_registry
    from src.utils.tracking import setup_mlflow

    setup_mlflow(params["mlflow"]["experiment_name"])
    name = params["register"]["model_name"]
    champion = load_from_registry(name, a.champion_stage)
    if champion is None:
        sys.exit(f"No model in stage {a.champion_stage}: promote one with scripts.promote")
    candidate = load_from_registry(name, a.candidate_stage)

    ev = fixed_test_set(a.n, a.seed)
    waves = [sf.read(ROOT / p, dtype="float32")[0] for p in ev["path"]]
    champ_eer = eer_of(champion, waves, ev["label"])
    cand_eer = eer_of(candidate, waves, ev["label"]) if candidate is not None else None
    result = decide_gate(champ_eer, cand_eer, a.tolerance, a.max_eer)
    result.update({"n_clips": len(ev), "seed": a.seed, "champion_version": champion.version,
                   "candidate_version": candidate.version if candidate is not None else None})
    save_json(result, REPORTS_DIR / "eval_gate.json")

    lines = ["### Model evaluation gate (fixed test set, unseen attacks)", "",
             "| model | version | EER |", "|---|---|---|",
             f"| champion ({a.champion_stage}) | v{champion.version} | {champ_eer:.4f} |"]
    if candidate is not None:
        lines.append(f"| challenger ({a.candidate_stage}) | v{candidate.version} | {cand_eer:.4f} |")
    else:
        lines.append(f"| challenger ({a.candidate_stage}) | none | - |")
    lines += ["", f"{len(ev)} clips, seed {a.seed}, tolerance {a.tolerance}, ceiling {a.max_eer}.",
              "", "**PASSED**" if result["passed"] else "**BLOCKED**: " + "; ".join(result["reasons"])]
    print("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
