"""Fake-speech generators for the continuous red-team.

* PiperGenerator - fast neural TTS (CPU friendly). Multi-speaker voices give variety.
  Voices are downloaded once:  python -m piper.download_voices en_US-libritts_r-medium --data-dir voices
* XTTSGenerator  - zero-shot VOICE CLONING from a 3-10 s reference clip (coqui-tts, GPU
  recommended e.g. Kaggle). This is the realistic scam scenario: clone a real speaker.
  Install: pip install coqui-tts   and set COQUI_TOS_AGREED=1 (non-commercial CPML licence)

All generators return float32 mono audio at 16 kHz.
"""
from __future__ import annotations

import io
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 16000


def _resample(wav: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR:
        return wav.astype(np.float32)
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(sr_in), "-ac", "1",
           "-i", "pipe:0", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, input=wav.astype(np.float32).tobytes(), capture_output=True,
                         check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


class Generator:
    name = "base"

    def generate(self, text: str, rng: np.random.Generator,
                 reference_wav: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


class PiperGenerator(Generator):
    name = "piper"

    def __init__(self, voice_paths: list[str]):
        from piper import PiperVoice

        if not voice_paths:
            raise ValueError("No Piper voices given (download with piper.download_voices)")
        self.voices = []
        for p in voice_paths:
            v = PiperVoice.load(p)
            cfg = json.loads(Path(f"{p}.json").read_text()) if Path(f"{p}.json").exists() else {}
            self.voices.append((Path(p).stem, v, int(cfg.get("num_speakers", 1))))

    def generate(self, text, rng, reference_wav=None):
        from piper import SynthesisConfig

        name, voice, n_spk = self.voices[int(rng.integers(len(self.voices)))]
        cfg = SynthesisConfig(speaker_id=int(rng.integers(n_spk)) if n_spk > 1 else None,
                              length_scale=float(rng.uniform(0.9, 1.15)),
                              noise_scale=float(rng.uniform(0.5, 0.8)))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf, syn_config=cfg)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        self.last_variant = name
        return _resample(pcm.astype(np.float32) / 32768.0, sr)


class XTTSGenerator(Generator):
    name = "xtts"

    def __init__(self, device: str = "cuda"):
        import os

        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS

        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def generate(self, text, rng, reference_wav=None):
        import tempfile

        import soundfile as sf

        if reference_wav is None:
            raise ValueError("XTTS voice cloning needs a reference clip")
        with tempfile.NamedTemporaryFile(suffix=".wav") as ref:
            sf.write(ref.name, reference_wav, SR)
            out = self.tts.tts(text=text, speaker_wav=ref.name, language="en")
        self.last_variant = "xtts_v2"
        return _resample(np.asarray(out, dtype=np.float32), 24000)


class StubGenerator(Generator):
    """Deterministic synthetic 'vocoder' - ONLY for tests/CI where no TTS model is available."""
    name = "stub"

    def generate(self, text, rng, reference_wav=None):
        t = np.arange(int(SR * (1.5 + 0.05 * len(text) % 2))) / SR
        f0 = rng.uniform(110, 220)
        x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
        x += 0.1 * np.sin(2 * np.pi * 3500 * t)
        self.last_variant = "stub"
        return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def build(names: list[str], voices_dir: str = "voices") -> list[Generator]:
    gens = []
    for n in names:
        if n == "piper":
            gens.append(PiperGenerator(sorted(str(p) for p in Path(voices_dir).glob("*.onnx"))))
        elif n == "xtts":
            gens.append(XTTSGenerator())
        elif n == "stub":
            gens.append(StubGenerator())
        else:
            raise ValueError(f"unknown generator {n}")
    return gens
