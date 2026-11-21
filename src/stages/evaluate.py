"""Stage 7 - evaluate: score the eval subset (UNSEEN attacks A07-A19) with both models.

Outputs DVC metrics/plots and logs everything to a dedicated MLflow evaluation run.
The dev-set EER threshold of each model is reused on eval (no peeking at eval labels).
"""
from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.stages.train_baseline import load_split
from src.stages.train_main import SpoofDataset, build_model, pick_device, score
from src.utils.common import (METADATA_CSV, MODELS_DIR, REPORTS_DIR, get_logger,
                              load_json, load_params, save_json, set_seed)
from src.utils.metrics import summarize
from src.utils.tracking import lineage_tags, setup_mlflow

log = get_logger("evaluate")


def main() -> None:
    params = load_params()
    set_seed(params["seed"])
    setup_mlflow(params["mlflow"]["experiment_name"])
    meta = pd.read_csv(METADATA_CSV)
    ev = meta[meta["split"] == "eval"].reset_index(drop=True)

    # ---- baseline ----
    with open(MODELS_DIR / "baseline" / "gmm.pkl", "rb") as f:
        gmm = pickle.load(f)
    b_meta = load_json(MODELS_DIR / "baseline" / "meta.json")
    feats, lengths, m_eval = load_split("eval", meta)
    b_scores = pd.Series(gmm.score_utterances(feats, lengths), index=m_eval.index)
    b_scores = b_scores.loc[ev["utt_id"]].to_numpy()

    # ---- main ----
    mp = params["main"]
    device = pick_device(mp["device"])
    m_meta = load_json(MODELS_DIR / "main" / "meta.json")
    model = build_model(mp["model_config"], device)
    model.load_state_dict(torch.load(MODELS_DIR / "main" / "aasist_best.pt", map_location=device))
    dl = DataLoader(SpoofDataset(ev, mp["nb_samp"], False, params["seed"]),
                    batch_size=mp["batch_size"], num_workers=mp["num_workers"])
    m_scores = score(model, dl, device)

    labels, attacks = ev["label"].to_numpy(), ev["attack_id"].to_numpy()
    results = {
        "baseline_lfcc_gmm": summarize(b_scores, labels, attacks, b_meta["dev_eer_threshold"]),
        "main_aasist_l": summarize(m_scores, labels, attacks, m_meta["dev_eer_threshold"]),
    }

    # ---- DVC metrics (flat) ----
    metrics = {name: {k: r[k] for k in ("eer", "auc", "accuracy")} for name, r in results.items()}
    save_json(metrics, REPORTS_DIR / "metrics.json")

    per_attack = pd.DataFrame({
        "attack_id": list(results["main_aasist_l"]["per_attack_eer"].keys()),
        "baseline_eer": list(results["baseline_lfcc_gmm"]["per_attack_eer"].values()),
        "main_eer": list(results["main_aasist_l"]["per_attack_eer"].values()),
    })
    per_attack.to_csv(REPORTS_DIR / "per_attack_eer.csv", index=False)

    # ---- ROC points for `dvc plots` ----
    is_bona = (labels == "bonafide").astype(int)
    roc_rows = []
    for name, s in (("baseline", b_scores), ("main", m_scores)):
        fpr, tpr, _ = roc_curve(is_bona, s)
        roc_rows += [{"model": name, "fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)]
    pd.DataFrame(roc_rows).to_csv(REPORTS_DIR / "roc.csv", index=False)

    # ---- figures ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(per_attack))
    ax[0].bar(x - 0.2, per_attack["baseline_eer"] * 100, 0.4, label="LFCC-GMM")
    ax[0].bar(x + 0.2, per_attack["main_eer"] * 100, 0.4, label="AASIST-L")
    ax[0].set_xticks(x, per_attack["attack_id"], rotation=45)
    ax[0].set(title="EER per unseen attack (eval)", ylabel="EER (%)")
    ax[0].legend()
    for name, s in (("LFCC-GMM", b_scores), ("AASIST-L", m_scores)):
        fpr, tpr, _ = roc_curve(is_bona, s)
        ax[1].plot(fpr, tpr, label=name)
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[1].set(title="ROC (bona fide = positive)", xlabel="FPR", ylabel="TPR")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eval_overview.png", dpi=120)
    plt.close(fig)

    # ---- MLflow: evaluation run + attach eval metrics to each training run ----
    client = mlflow.tracking.MlflowClient()
    for name, run_id in (("baseline_lfcc_gmm", b_meta["run_id"]),
                         ("main_aasist_l", m_meta["run_id"])):
        r = results[name]
        for k in ("eer", "auc", "accuracy"):
            client.log_metric(run_id, f"eval_{k}", r[k])

    with mlflow.start_run(run_name="evaluation-unseen-attacks"):
        mlflow.set_tags({**lineage_tags(), "model_type": "evaluation",
                         "baseline_run_id": b_meta["run_id"], "main_run_id": m_meta["run_id"]})
        mlflow.log_params({"n_eval": len(ev), "eval_attacks": ",".join(sorted(set(attacks) - {"-"}))})
        for name, r in results.items():
            mlflow.log_metrics({f"{name}.eer": r["eer"], f"{name}.auc": r["auc"],
                                f"{name}.accuracy": r["accuracy"]})
            mlflow.log_metrics({f"{name}.eer_{a}": v for a, v in r["per_attack_eer"].items()})
        for f in ("metrics.json", "per_attack_eer.csv", "eval_overview.png", "validation.json",
                  "data_summary.json"):
            mlflow.log_artifact(str(REPORTS_DIR / f))
        save_json(results, REPORTS_DIR / "eval_full.json")
        mlflow.log_artifact(str(REPORTS_DIR / "eval_full.json"))

    log.info("Eval EER - baseline: %.4f | main: %.4f",
             results["baseline_lfcc_gmm"]["eer"], results["main_aasist_l"]["eer"])


if __name__ == "__main__":
    main()
