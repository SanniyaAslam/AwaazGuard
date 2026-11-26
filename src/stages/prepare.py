"""Stage 3 - prepare: build a reproducible, stratified subset and a single metadata table.

* Bona fide / spoof ratio is fixed by `prepare.bonafide_fraction`.
* Spoofed utterances are spread evenly over every attack ID in the split, so the eval
  subset contains all 13 unseen attacks (A07-A19) for per-attack EER.
* Selected FLAC files are copied to data/processed/audio/<split>/ - this folder plus
  metadata.csv is the dataset version tracked by DVC and pushed to the DagsHub remote.
* Labelled production feedback (data/feedback, exported by scripts/export_feedback.py and
  versioned with `dvc add`) is appended to the TRAIN split - this closes the MLOps loop.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.stages.validate import read_protocol
from src.utils.common import (LABELS, METADATA_CSV, PROCESSED_DIR, REPORTS_DIR, SPLITS, ROOT,
                              audio_path, get_logger, load_params, save_json)

log = get_logger("prepare")


def stratified_subset(df: pd.DataFrame, n_total: int, bona_frac: float,
                      rng: np.random.Generator) -> pd.DataFrame:
    bona = df[df["label"] == "bonafide"]
    spoof = df[df["label"] == "spoof"]
    n_bona = min(int(round(n_total * bona_frac)), len(bona))
    n_spoof = min(n_total - n_bona, len(spoof))

    parts = [bona.sample(n=n_bona, random_state=int(rng.integers(1 << 31)))]
    attacks = sorted(spoof["attack_id"].unique())
    base, extra = divmod(n_spoof, len(attacks))
    for i, atk in enumerate(attacks):
        pool = spoof[spoof["attack_id"] == atk]
        k = min(base + (1 if i < extra else 0), len(pool))
        parts.append(pool.sample(n=k, random_state=int(rng.integers(1 << 31))))
    return pd.concat(parts).sort_values("utt_id").reset_index(drop=True)


def add_feedback(p: dict, rng: np.random.Generator) -> list[dict]:
    """Append labelled production clips (data/feedback/labels.csv) to the train split."""
    fb_dir = ROOT / p.get("feedback_dir", "data/feedback")
    labels_csv = fb_dir / "labels.csv"
    if not labels_csv.exists():
        return []
    fb = pd.read_csv(labels_csv)
    fb = fb[fb["label"].isin(LABELS) & fb["file"].map(lambda f: (fb_dir / f).exists())]
    if p.get("max_feedback") and len(fb) > p["max_feedback"]:
        fb = fb.sample(n=p["max_feedback"], random_state=int(rng.integers(1 << 31)))
    out_dir = PROCESSED_DIR / "audio" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in fb.itertuples():
        dst = out_dir / f"FB_{r.prediction_id}{Path(r.file).suffix}"
        shutil.copy2(fb_dir / r.file, dst)
        info = sf.info(str(dst))
        rows.append({
            "utt_id": f"FB_{r.prediction_id}", "speaker_id": "FEEDBACK", "split": "train",
            "attack_id": "-" if r.label == "bonafide" else "PROD", "label": r.label,
            "path": dst.relative_to(ROOT).as_posix(),
            "duration_s": round(info.frames / info.samplerate, 3),
        })
    log.info("Added %d feedback clips to train", len(rows))
    return rows


def main() -> None:
    params = load_params()
    p = params["prepare"]
    rng = np.random.default_rng(params["seed"])
    sizes = {"train": p["n_train"], "dev": p["n_dev"], "eval": p["n_eval"]}

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)          # rebuild from scratch every time
    rows = []
    for split in SPLITS:
        sub = stratified_subset(read_protocol(split), sizes[split], p["bonafide_fraction"], rng)
        out_dir = PROCESSED_DIR / "audio" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in tqdm(sub.itertuples(), total=len(sub), desc=f"copy {split}"):
            src = audio_path(split, r.utt_id)
            dst = out_dir / f"{r.utt_id}.flac"
            shutil.copy2(src, dst)
            info = sf.info(str(dst))
            rows.append({
                "utt_id": r.utt_id, "speaker_id": r.speaker_id, "split": split,
                "attack_id": r.attack_id, "label": r.label,
                "path": dst.relative_to(ROOT).as_posix(),
                "duration_s": round(info.frames / info.samplerate, 3),
            })

    rows += add_feedback(p, rng)
    meta = pd.DataFrame(rows)
    meta.to_csv(METADATA_CSV, index=False)

    summary = {
        split: {
            "utterances": int((meta["split"] == split).sum()),
            "bonafide": int(((meta["split"] == split) & (meta["label"] == "bonafide")).sum()),
            "spoof": int(((meta["split"] == split) & (meta["label"] == "spoof")).sum()),
            "speakers": int(meta.loc[meta["split"] == split, "speaker_id"].nunique()),
            "attacks": sorted(meta.loc[(meta["split"] == split) & (meta["label"] == "spoof"),
                                       "attack_id"].unique().tolist()),
            "hours": round(float(meta.loc[meta["split"] == split, "duration_s"].sum()) / 3600, 3),
        }
        for split in SPLITS
    }
    save_json(summary, REPORTS_DIR / "data_summary.json")
    log.info("Subset written to %s: %s", PROCESSED_DIR,
             {s: v["utterances"] for s, v in summary.items()})


if __name__ == "__main__":
    main()
