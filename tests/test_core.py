"""Unit tests for the deterministic building blocks (run with `pytest -q`)."""
import numpy as np
import torch

from src.features.lfcc import lfcc
from src.models.aasist import Model as AASIST
from src.utils.audio import pad_or_crop
from src.utils.common import load_params
from src.utils.metrics import compute_eer, summarize


def test_eer_perfect_separation():
    eer, _ = compute_eer(np.array([2.0, 3.0, 4.0]), np.array([-1.0, 0.0, 1.0]))
    assert eer == 0.0


def test_eer_random_is_about_half():
    rng = np.random.default_rng(0)
    eer, _ = compute_eer(rng.normal(size=5000), rng.normal(size=5000))
    assert 0.45 < eer < 0.55


def test_summarize_per_attack():
    scores = np.array([3, 3, -3, -3, 0.5])
    labels = np.array(["bonafide", "bonafide", "spoof", "spoof", "spoof"])
    attacks = np.array(["-", "-", "A07", "A07", "A08"])
    r = summarize(scores, labels, attacks)
    assert set(r["per_attack_eer"]) == {"A07", "A08"}
    assert r["per_attack_eer"]["A07"] == 0.0


def test_lfcc_shape_and_determinism():
    wav = np.random.default_rng(1).standard_normal(16000).astype(np.float32)
    f1, f2 = lfcc(wav, 16000), lfcc(wav, 16000)
    assert f1.shape == (1 + (16000 - 320) // 160, 60)
    assert np.allclose(f1, f2)
    assert np.isfinite(f1).all()


def test_pad_or_crop():
    assert len(pad_or_crop(np.ones(100, np.float32), 1000)) == 1000
    assert len(pad_or_crop(np.ones(5000, np.float32), 1000)) == 1000


def test_aasist_forward_shape():
    cfg = load_params()["main"]["model_config"]
    model = AASIST(cfg).eval()
    with torch.no_grad():
        _, out = model(torch.randn(2, cfg["nb_samp"]))
    assert out.shape == (2, 2)
