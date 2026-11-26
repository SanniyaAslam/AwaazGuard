"""Audio decoding (any format -> 16 kHz mono float32 via ffmpeg) and sliding windows."""
from __future__ import annotations

import io
import subprocess

import numpy as np
import soundfile as sf

SR = 16000


class AudioError(ValueError):
    pass


def decode_bytes(data: bytes, max_seconds: float = 60.0) -> np.ndarray:
    """Decode WAV/MP3/OGG/OPUS/WEBM/M4A... to 16 kHz mono float32 using ffmpeg."""
    if not data:
        raise AudioError("Empty upload")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-t", str(max_seconds),
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, input=data, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise AudioError(f"Could not decode audio: {proc.stderr.decode(errors='ignore')[:200]}")
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if len(wav) < SR * 0.3:
        raise AudioError("Audio is shorter than 0.3 s")
    return wav


def to_wav_bytes(wav: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def trim_silence(wav: np.ndarray, frame: int = 400, rel_db: float = -40.0) -> np.ndarray:
    """Simple energy-based VAD: drop leading/trailing frames 40 dB below the peak frame."""
    n = len(wav) // frame
    if n < 3:
        return wav
    energy = np.sqrt(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(energy / energy.max())
    voiced = np.where(db > rel_db)[0]
    if len(voiced) == 0:
        return wav
    return wav[voiced[0] * frame:(voiced[-1] + 1) * frame]


def make_windows(wav: np.ndarray, win: int, hop: int) -> tuple[np.ndarray, list[int]]:
    """Fixed-length windows (repeat-padded like AASIST training) and their start samples."""
    if len(wav) <= win:
        reps = int(np.ceil(win / max(len(wav), 1)))
        return np.tile(wav, reps)[:win][None, :].astype(np.float32), [0]
    starts = list(range(0, len(wav) - win + 1, hop))
    if starts[-1] + win < len(wav):
        starts.append(len(wav) - win)
    return np.stack([wav[s:s + win] for s in starts]).astype(np.float32), starts


def audio_stats(wav: np.ndarray) -> dict:
    return {
        "duration_s": round(len(wav) / SR, 3),
        "rms": float(np.sqrt(np.mean(wav ** 2))),
        "peak": float(np.max(np.abs(wav))) if len(wav) else 0.0,
        "clipped_fraction": float(np.mean(np.abs(wav) >= 0.999)) if len(wav) else 0.0,
    }
