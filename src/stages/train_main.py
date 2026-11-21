"""Stage 6 - train_main: AASIST-L on raw waveforms, logged to MLflow every epoch.

The best epoch (lowest dev EER) is saved to models/main/ and logged as an MLflow
PyTorch model so the `register` stage can push it to the Model Registry.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.aasist import Model as AASIST
from src.utils.audio import load_audio, pad_or_crop
from src.utils.common import (METADATA_CSV, MODELS_DIR, REPORTS_DIR, ROOT, get_logger,
                              load_params, save_json, set_seed)
from src.utils.metrics import summarize
from src.utils.tracking import lineage_tags, setup_mlflow

log = get_logger("train_main")
OUT_DIR = MODELS_DIR / "main"


class SpoofDataset(Dataset):
    """Label convention (AASIST): 1 = bona fide, 0 = spoof."""

    def __init__(self, meta: pd.DataFrame, nb_samp: int, train: bool, seed: int):
        self.meta = meta.reset_index(drop=True)
        self.nb_samp, self.train = nb_samp, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, i: int):
        row = self.meta.iloc[i]
        wav, _ = load_audio(ROOT / row["path"])
        x = pad_or_crop(wav, self.nb_samp, random_crop=self.train, rng=self.rng)
        return torch.from_numpy(x), int(row["label"] == "bonafide")


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def score(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        _, logits = model(x.to(device))
        out.append(logits[:, 1].float().cpu().numpy())   # bona fide logit = score
    return np.concatenate(out)


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    return AASIST(cfg).to(device)


def main() -> None:
    params = load_params()
    mp, seed = params["main"], params["seed"]
    set_seed(seed)
    device = pick_device(mp["device"])
    setup_mlflow(params["mlflow"]["experiment_name"])
    meta = pd.read_csv(METADATA_CSV)

    g = torch.Generator().manual_seed(seed)
    train_ds = SpoofDataset(meta[meta["split"] == "train"], mp["nb_samp"], True, seed)
    dev_meta = meta[meta["split"] == "dev"].reset_index(drop=True)
    dev_ds = SpoofDataset(dev_meta, mp["nb_samp"], False, seed)
    train_dl = DataLoader(train_ds, batch_size=mp["batch_size"], shuffle=True,
                          num_workers=mp["num_workers"], generator=g, drop_last=True)
    dev_dl = DataLoader(dev_ds, batch_size=mp["batch_size"], shuffle=False,
                        num_workers=mp["num_workers"])

    model = build_model(mp["model_config"], device)
    n_params = sum(p.numel() for p in model.parameters())
    optim = torch.optim.Adam(model.parameters(), lr=mp["lr"], weight_decay=mp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(mp["epochs"], 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(mp["class_weights"], device=device))

    history, best = [], {"eer": float("inf"), "epoch": -1}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="main-aasist-l") as run:
        mlflow.set_tags({**lineage_tags(), "model_type": "main", "model": "AASIST-L",
                         "device": str(device)})
        flat = {k: v for k, v in mp.items() if k != "model_config"}
        mlflow.log_params({f"main.{k}": v for k, v in flat.items()})
        mlflow.log_params({f"arch.{k}": str(v) for k, v in mp["model_config"].items()})
        mlflow.log_params({"seed": seed, "n_parameters": n_params,
                           "n_train": len(train_ds), "n_dev": len(dev_ds)})

        t_start = time.time()
        for epoch in range(1, mp["epochs"] + 1):
            model.train()
            losses, t0 = [], time.time()
            for b, (x, y) in enumerate(train_dl):
                if mp["max_batches_per_epoch"] and b >= mp["max_batches_per_epoch"]:
                    break
                _, logits = model(x.to(device), Freq_aug=True)
                loss = criterion(logits, y.to(device))
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
            sched.step()

            dev_scores = score(model, dev_dl, device)
            dev = summarize(dev_scores, dev_meta["label"].to_numpy(),
                            dev_meta["attack_id"].to_numpy())
            rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                   "dev_eer": dev["eer"], "dev_auc": dev["auc"],
                   "lr": optim.param_groups[0]["lr"], "epoch_time_s": time.time() - t0}
            history.append(rec)
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=epoch)
            log.info("epoch %d | loss %.4f | dev EER %.4f", epoch, rec["train_loss"], dev["eer"])

            if dev["eer"] < best["eer"]:
                best = {"eer": dev["eer"], "epoch": epoch, "threshold": dev["eer_threshold"],
                        "auc": dev["auc"]}
                torch.save(model.state_dict(), OUT_DIR / "aasist_best.pt")

        # ---- reload best epoch, log it as an MLflow model ----
        model.load_state_dict(torch.load(OUT_DIR / "aasist_best.pt", map_location=device))
        model.eval().cpu()
        mlflow.log_metrics({"best_dev_eer": best["eer"], "best_epoch": best["epoch"],
                            "best_dev_auc": best["auc"],
                            "total_train_time_s": time.time() - t_start})
        mlflow.pytorch.log_model(model, artifact_path="model",
                                 pip_requirements=["torch", "numpy"])

        hist = pd.DataFrame(history)
        hist.to_csv(REPORTS_DIR / "main_training_history.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(hist["epoch"], hist["train_loss"], marker="o")
        ax[0].set(title="Training loss", xlabel="epoch")
        ax[1].plot(hist["epoch"], hist["dev_eer"] * 100, marker="o", color="tab:red")
        ax[1].set(title="Dev EER (%)", xlabel="epoch")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "main_learning_curves.png", dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(REPORTS_DIR / "main_learning_curves.png"))
        mlflow.log_artifact(str(REPORTS_DIR / "main_training_history.csv"))
        mlflow.log_artifact(str(ROOT / "params.yaml"))

        save_json({"run_id": run.info.run_id, "best_epoch": best["epoch"],
                   "dev_eer_threshold": best["threshold"], "n_parameters": n_params},
                  OUT_DIR / "meta.json")
        save_json({"best_dev_eer": best["eer"], "best_epoch": best["epoch"],
                   "n_parameters": n_params}, REPORTS_DIR / "main_train.json")
        log.info("Best dev EER %.4f at epoch %d (run %s)", best["eer"], best["epoch"],
                 run.info.run_id)


if __name__ == "__main__":
    main()
