"""Stage 4 - featurize: extract LFCC frames for every utterance (baseline front-end).

Saves one compressed .npz per split with all frames concatenated plus per-utterance
offsets, so the baseline can fit GMMs on frames and score utterances quickly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.lfcc import lfcc
from src.utils.audio import load_audio
from src.utils.common import (FEATURES_DIR, METADATA_CSV, ROOT, SPLITS, get_logger,
                              load_params)

log = get_logger("featurize")


def main() -> None:
    fp = load_params()["features"]
    meta = pd.read_csv(METADATA_CSV)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        sub = meta[meta["split"] == split]
        feats, lengths = [], []
        for path in tqdm(sub["path"], desc=f"LFCC {split}"):
            wav, sr = load_audio(ROOT / path)
            f = lfcc(wav, sr, **fp)
            feats.append(f.astype(np.float16))
            lengths.append(len(f))
        np.savez_compressed(
            FEATURES_DIR / f"lfcc_{split}.npz",
            feats=np.concatenate(feats),
            lengths=np.asarray(lengths, dtype=np.int32),
            utt_ids=sub["utt_id"].to_numpy(),
        )
        log.info("%s: %d utterances, %d frames, dim=%d", split, len(lengths),
                 sum(lengths), feats[0].shape[1])


if __name__ == "__main__":
    main()
