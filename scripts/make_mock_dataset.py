"""Create a tiny synthetic dataset with the exact ASVspoof 2019 LA folder/protocol layout.

Used ONLY for fast smoke tests of the pipeline (no 7 GB download). Bona fide clips are
harmonic tones + noise; spoof clips add attack-specific spectral artefacts.

    python scripts/make_mock_dataset.py --out mock_data/LA --per-split 120
    # then set ingest.local_dir: mock_data/LA in params.yaml (or `dvc exp run -S`)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
PREFIX = {"train": "T", "dev": "D", "eval": "E"}
ATTACKS = {"train": range(1, 7), "dev": range(1, 7), "eval": range(7, 20)}
SPEAKER_BASE = {"train": 1000, "dev": 3000, "eval": 6000}
PROTO = {"train": "trn", "dev": "trl", "eval": "trl"}


def synth(rng, spoof: bool, attack: int) -> np.ndarray:
    dur = rng.uniform(1.5, 4.5)
    t = np.arange(int(dur * SR)) / SR
    f0 = rng.uniform(100, 250)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 5) * t))
    x += 0.05 * rng.standard_normal(len(t))
    if spoof:
        x += 0.15 * np.sin(2 * np.pi * (3000 + 150 * attack) * t)   # vocoder-like tone
        x = np.round(x * 8) / 8 * 0.9 + 0.1 * x                     # quantisation artefact
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mock_data/LA")
    ap.add_argument("--per-split", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out)
    (root / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "eval"):
        flac_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
        flac_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        attacks = list(ATTACKS[split])
        for i in range(a.per_split):
            utt = f"LA_{PREFIX[split]}_{1000000 + i:07d}"
            spk = f"LA_{SPEAKER_BASE[split] + i % 8:04d}"
            spoof = i % 3 != 0
            atk = attacks[i % len(attacks)] if spoof else 0
            sf.write(flac_dir / f"{utt}.flac", synth(rng, spoof, atk), SR)
            lines.append(f"{spk} {utt} - {'A%02d' % atk if spoof else '-'} "
                         f"{'spoof' if spoof else 'bonafide'}")
        (root / "ASVspoof2019_LA_cm_protocols" /
         f"ASVspoof2019.LA.cm.{split}.{PROTO[split]}.txt").write_text("\n".join(lines) + "\n")
    print(f"Mock dataset written to {root}")


if __name__ == "__main__":
    main()
