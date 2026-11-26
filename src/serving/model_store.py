"""Load the ONNX detector either from the MLflow Model Registry (by stage) or a local folder.

Registry flow: registered model `awaazguard-aasist`, stage `Production` (champion) or
`Staging` (candidate) -> model version -> source run -> artifacts `serving/model.onnx`
and `serving/reference.npz` (written by the export_onnx DVC stage).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.serving.audio import make_windows


@dataclass
class LoadedModel:
    session: ort.InferenceSession
    threshold: float
    nb_samp: int
    version: str
    source: str
    ref_embeddings: np.ndarray = field(repr=False, default=None)
    ref_margins: np.ndarray = field(repr=False, default=None)

    def predict(self, wav: np.ndarray, hop_s: float = 1.0, batch: int = 4, sr: int = 16000) -> dict:
        """Score a clip with overlapping windows. Returns clip + per-second results."""
        windows, starts = make_windows(wav, self.nb_samp, int(hop_s * sr))
        embs, margins = [], []
        for i in range(0, len(windows), batch):
            e, logits = self.session.run(None, {"wave": windows[i:i + batch]})
            embs.append(e)
            margins.append(logits[:, 1] - logits[:, 0])
        embs, margins = np.concatenate(embs), np.concatenate(margins)

        # calibrated so prob_fake = 0.5 exactly at the dev-set EER threshold
        def p_fake(m):
            return 1.0 / (1.0 + np.exp(np.clip(m - self.threshold, -30, 30)))

        clip_margin = float(margins.mean())
        n_sec = max(1, int(np.ceil(len(wav) / sr)))
        seg = []
        for s in range(n_sec):
            a, b = s * sr, (s + 1) * sr
            cover = [m for st, m in zip(starts, margins) if st < b and st + self.nb_samp > a]
            m = float(np.mean(cover)) if cover else clip_margin
            seg.append({"start": s, "end": min(s + 1, round(len(wav) / sr, 2)),
                        "prob_fake": round(float(p_fake(m)), 4)})
        return {
            "margin": clip_margin,
            "prob_fake": float(p_fake(clip_margin)),
            "segments": seg,
            "embedding": embs.mean(axis=0).astype(np.float32),
            "n_windows": int(len(windows)),
        }


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
    if os.getenv("ORT_LOW_MEMORY", "1") == "1":   # return memory after each request (1 GB VMs)
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def load_from_dir(folder: str | Path, version: str = "local") -> LoadedModel:
    folder = Path(folder)
    ref = np.load(folder / "reference.npz")
    return LoadedModel(session=_session(folder / "model.onnx"), threshold=float(ref["threshold"]),
                       nb_samp=int(ref["nb_samp"]), version=version, source=str(folder),
                       ref_embeddings=ref["embeddings"], ref_margins=ref["margins"])


def load_from_registry(model_name: str, stage: str) -> LoadedModel | None:
    """Return the latest version in `stage`, or None if that stage is empty."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return None
    mv = versions[0]
    dst = Path(tempfile.mkdtemp(prefix=f"model_v{mv.version}_"))
    client.download_artifacts(mv.run_id, "serving", str(dst))
    m = load_from_dir(dst / "serving", version=str(mv.version))
    m.source = f"models:/{model_name}/{stage} (v{mv.version}, run {mv.run_id[:8]})"
    return m
