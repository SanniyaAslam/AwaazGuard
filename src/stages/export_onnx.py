"""Stage - export_onnx: convert the best AASIST-L checkpoint to ONNX for serving and build the
reference statistics the production system needs.

Outputs (models/serving/):
  model.onnx      - inputs: wave [batch, nb_samp]; outputs: embedding [batch,160], logits [batch,2]
  reference.npz   - dev-set embeddings, margin scores, labels and the decision threshold
                    (used by the API for probabilities and by the drift monitor as baseline)
Both files are also logged to the main MLflow run under `serving/`, which is where the
inference API downloads them from when it loads a model by registry stage.
"""
from __future__ import annotations

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.stages.train_main import SpoofDataset, build_model
from src.utils.common import (METADATA_CSV, MODELS_DIR, REPORTS_DIR, get_logger, load_json,
                              load_params, save_json, set_seed)
from src.utils.metrics import compute_eer
from src.utils.tracking import setup_mlflow

log = get_logger("export_onnx")
OUT_DIR = MODELS_DIR / "serving"


def main() -> None:
    params = load_params()
    mp = params["main"]
    set_seed(params["seed"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(mp["model_config"], torch.device("cpu"))
    model.load_state_dict(torch.load(MODELS_DIR / "main" / "aasist_best.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(2, mp["nb_samp"])
    onnx_path = OUT_DIR / "model.onnx"
    torch.onnx.export(model, (dummy,), str(onnx_path), input_names=["wave"],
                      output_names=["embedding", "logits"], opset_version=17,
                      dynamic_axes={"wave": {0: "batch"}, "embedding": {0: "batch"},
                                    "logits": {0: "batch"}}, dynamo=False)

    # ---- parity check: ONNX must match PyTorch ----
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        _, ref_logits = model(dummy)
    _, onnx_logits = sess.run(None, {"wave": dummy.numpy()})
    max_diff = float(np.abs(onnx_logits - ref_logits.numpy()).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {max_diff}")

    # ---- reference statistics on the dev set (known attacks, never used for training) ----
    meta = pd.read_csv(METADATA_CSV)
    dev = meta[meta["split"] == "dev"].reset_index(drop=True)
    dl = DataLoader(SpoofDataset(dev, mp["nb_samp"], False, params["seed"]),
                    batch_size=4, num_workers=0)
    embs, margins = [], []
    for x, _ in dl:
        e, logit = sess.run(None, {"wave": x.numpy()})
        embs.append(e)
        margins.append(logit[:, 1] - logit[:, 0])          # log-odds bona fide vs spoof
    embs, margins = np.concatenate(embs), np.concatenate(margins)
    labels = (dev["label"] == "bonafide").to_numpy().astype(np.int8)
    eer, thr = compute_eer(margins[labels == 1], margins[labels == 0])
    np.savez_compressed(OUT_DIR / "reference.npz", embeddings=embs.astype(np.float32),
                        margins=margins.astype(np.float32), labels=labels,
                        threshold=np.float32(thr), nb_samp=np.int32(mp["nb_samp"]))

    info = {"onnx_max_abs_diff": max_diff, "reference_size": int(len(dev)),
            "dev_eer_margin": eer, "threshold": thr,
            "onnx_mb": round(onnx_path.stat().st_size / 1e6, 3)}
    save_json(info, REPORTS_DIR / "export.json")

    setup_mlflow(params["mlflow"]["experiment_name"])
    run_id = load_json(MODELS_DIR / "main" / "meta.json")["run_id"]
    client = mlflow.tracking.MlflowClient()
    client.log_artifact(run_id, str(onnx_path), artifact_path="serving")
    client.log_artifact(run_id, str(OUT_DIR / "reference.npz"), artifact_path="serving")
    client.log_metric(run_id, "onnx_max_abs_diff", max_diff)
    log.info("ONNX exported (%.3f MB, max diff %.2e), threshold %.4f", info["onnx_mb"],
             max_diff, thr)


if __name__ == "__main__":
    main()
