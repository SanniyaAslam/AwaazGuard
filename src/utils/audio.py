"""Audio loading and fixed-length padding/cropping."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:  # down-mix to mono if ever needed
        wav = wav.mean(axis=1)
    return wav, sr


def pad_or_crop(wav: np.ndarray, length: int, random_crop: bool = False,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Repeat-pad short clips (AASIST convention); crop long ones."""
    n = len(wav)
    if n == 0:
        return np.zeros(length, dtype=np.float32)
    if n < length:
        reps = int(np.ceil(length / n))
        return np.tile(wav, reps)[:length].astype(np.float32)
    if random_crop:
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, n - length + 1))
    else:
        start = 0
    return wav[start:start + length].astype(np.float32)
