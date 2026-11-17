"""Shared helpers: params loading, paths, seeding, logging, JSON IO."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Canonical locations used by every stage (mirrors dvc.yaml)
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
METADATA_CSV = PROCESSED_DIR / "metadata.csv"

SPLITS = ("train", "dev", "eval")
PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
# Attacks present in each official split (eval = unseen attacks, except A16=A04, A19=A06)
EXPECTED_ATTACKS = {
    "train": {f"A{i:02d}" for i in range(1, 7)},
    "dev": {f"A{i:02d}" for i in range(1, 7)},
    "eval": {f"A{i:02d}" for i in range(7, 20)},
}
LABELS = ("bonafide", "spoof")


def load_params(path: Path | None = None) -> dict[str, Any]:
    with open(path or ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def la_root() -> Path:
    """Folder that contains the extracted ASVspoof 2019 LA data."""
    return RAW_DIR / "LA"


def protocol_path(split: str) -> Path:
    return la_root() / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILES[split]


def audio_path(split: str, utt_id: str) -> Path:
    return la_root() / f"ASVspoof2019_LA_{split}" / "flac" / f"{utt_id}.flac"


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
