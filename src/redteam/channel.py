"""Real-world channel simulation: phone line, WhatsApp-style Opus, MP3 and background noise.

Used for (a) red-team batches and (b) optionally augmenting training data, so the detector
is tested on audio that looks like real scam calls / voice notes, not clean studio files.
"""
from __future__ import annotations

import subprocess

import numpy as np

SR = 16000
CHANNELS = ("clean", "phone", "opus", "mp3", "noise")


def _ffmpeg_roundtrip(wav: np.ndarray, encode_args: list[str], fmt: str,
                      pre_filter: str | None = None) -> np.ndarray:
    raw = wav.astype(np.float32).tobytes()
    enc = ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0"]
    if pre_filter:
        enc += ["-af", pre_filter]
    enc += encode_args + ["-f", fmt, "pipe:1"]
    coded = subprocess.run(enc, input=raw, capture_output=True, check=True).stdout
    dec = ["ffmpeg", "-loglevel", "error", "-f", fmt, "-i", "pipe:0", "-ac", "1", "-ar", str(SR),
           "-f", "f32le", "pipe:1"]
    out = np.frombuffer(subprocess.run(dec, input=coded, capture_output=True, check=True).stdout,
                        dtype=np.float32)
    return out.copy()


def phone(wav: np.ndarray) -> np.ndarray:
    """Narrow-band telephone: 300-3400 Hz, 8 kHz, mu-law (G.711)."""
    return _ffmpeg_roundtrip(wav, ["-ar", "8000", "-c:a", "pcm_mulaw"], "wav",
                             pre_filter="highpass=f=300,lowpass=f=3400")


def opus(wav: np.ndarray, kbps: int = 16) -> np.ndarray:
    """WhatsApp voice notes use Opus at low bit-rates."""
    return _ffmpeg_roundtrip(wav, ["-c:a", "libopus", "-b:a", f"{kbps}k"], "ogg")


def mp3(wav: np.ndarray, kbps: int = 32) -> np.ndarray:
    return _ffmpeg_roundtrip(wav, ["-c:a", "libmp3lame", "-b:a", f"{kbps}k"], "mp3")


def add_noise(wav: np.ndarray, rng: np.random.Generator, snr_db: tuple = (10, 25)) -> np.ndarray:
    snr = rng.uniform(*snr_db)
    noise = rng.standard_normal(len(wav)).astype(np.float32)
    noise = np.cumsum(noise) if rng.random() < 0.5 else noise      # brown-ish or white
    noise -= noise.mean()
    p_sig = np.mean(wav ** 2) + 1e-12
    p_noise = np.mean(noise ** 2) + 1e-12
    return (wav + noise * np.sqrt(p_sig / (p_noise * 10 ** (snr / 10)))).astype(np.float32)


def apply(wav: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "clean":
        out = wav
    elif kind == "phone":
        out = phone(wav)
    elif kind == "opus":
        out = opus(wav)
    elif kind == "mp3":
        out = mp3(wav)
    elif kind == "noise":
        out = add_noise(wav, rng)
    else:
        raise ValueError(f"unknown channel {kind}")
    peak = np.max(np.abs(out)) or 1.0
    return (out / max(peak, 1.0)).astype(np.float32)          # avoid clipping
