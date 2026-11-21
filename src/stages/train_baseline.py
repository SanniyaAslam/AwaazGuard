"""Stage 5 - train_baseline: LFCC + GMM, logged to MLflow."""
from __future__ import annotations

import pickle
import time

import mlflow
import numpy as np
import pandas as pd

from src.models.gmm import LfccGmmBaseline
from src.utils.common import (FEATURES_DIR, METADATA_CSV, MODELS_DIR, REPORTS_DIR, ROOT,
                              get_logger, load_params, save_json, set_seed)
from src.utils.metrics import summarize
from src.utils.tracking import lineage_tags, setup_mlflow

log = get_logger("train_baseline")
OUT_DIR = MODELS_DIR / "baseline"


def load_split(split: str, meta: pd.DataFrame):
    z = np.load(FEATURES_DIR / f"lfcc_{split}.npz", allow_pickle=True)
    m = meta.set_index("utt_id").loc[z["utt_ids"]]
    return z["feats"], z["lengths"], m


def frames_for(label_mask: np.ndarray, feats: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    frame_mask = np.repeat(label_mask, lengths)
    return feats[frame_mask]


def main() -> None:
    params = load_params()
    bp, seed = params["baseline"], params["seed"]
    set_seed(seed)
    setup_mlflow(params["mlflow"]["experiment_name"])
    meta = pd.read_csv(METADATA_CSV)

    feats, lengths, m_train = load_split("train", meta)
    is_bona = (m_train["label"] == "bonafide").to_numpy()
    rng = np.random.default_rng(seed)
    bona = frames_for(is_bona, feats, lengths)
    spoof = frames_for(~is_bona, feats, lengths)
    cap = bp["max_frames_per_class"]
    bona = bona[rng.choice(len(bona), min(cap, len(bona)), replace=False)]
    spoof = spoof[rng.choice(len(spoof), min(cap, len(spoof)), replace=False)]

    with mlflow.start_run(run_name="baseline-lfcc-gmm") as run:
        mlflow.set_tags({**lineage_tags(), "model_type": "baseline", "model": "LFCC-GMM"})
        mlflow.log_params({f"baseline.{k}": v for k, v in bp.items()})
        mlflow.log_params({f"features.{k}": v for k, v in params["features"].items()})
        mlflow.log_params({"seed": seed, "train_bona_frames": len(bona),
                           "train_spoof_frames": len(spoof)})

        t0 = time.time()
        model = LfccGmmBaseline(bp["n_components"], bp["covariance_type"], bp["max_iter"],
                                bp["reg_covar"], seed).fit(bona, spoof)
        train_time = time.time() - t0
        log.info("GMMs fitted in %.1fs", train_time)

        d_feats, d_len, m_dev = load_split("dev", meta)
        dev_scores = model.score_utterances(d_feats, d_len)
        dev = summarize(dev_scores, m_dev["label"].to_numpy(), m_dev["attack_id"].to_numpy())
        mlflow.log_metrics({"train_time_s": train_time, "dev_eer": dev["eer"],
                            "dev_auc": dev["auc"], "dev_accuracy": dev["accuracy"],
                            "gmm_bona_converged": float(model.gmm_bona.converged_),
                            "gmm_spoof_converged": float(model.gmm_spoof.converged_)})

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "gmm.pkl", "wb") as f:
            pickle.dump(model, f)
        save_json({"dev_eer_threshold": dev["eer_threshold"], "run_id": run.info.run_id},
                  OUT_DIR / "meta.json")
        mlflow.log_artifact(str(OUT_DIR / "gmm.pkl"), artifact_path="model")
        mlflow.log_artifact(str(ROOT / "params.yaml"))
        save_json({"dev_eer": dev["eer"], "dev_auc": dev["auc"],
                   "dev_accuracy": dev["accuracy"], "train_time_s": train_time},
                  REPORTS_DIR / "baseline_train.json")
        log.info("Baseline dev EER = %.4f (run %s)", dev["eer"], run.info.run_id)


if __name__ == "__main__":
    main()
