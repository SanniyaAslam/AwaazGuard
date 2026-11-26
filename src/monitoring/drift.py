"""Attack / domain drift detection on production traffic.

Three complementary signals, computed on the champion model's embeddings:
  1. MMD two-sample test (RBF kernel, permutation p-value): has the embedding distribution moved?
  2. Mahalanobis OOD rate: share of production clips further from the reference than the
     reference's own 99th percentile (looks like an unknown generator / channel).
  3. Evidently DataDriftPreset over the score and the top PCA components (HTML report).
Reference = dev-set embeddings saved by the export_onnx stage (shipped with every model).
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from src.utils.common import get_logger

log = get_logger("drift")
MMD_ALPHA = float(os.getenv("DRIFT_MMD_ALPHA", "0.01"))
OOD_LIMIT = float(os.getenv("DRIFT_OOD_LIMIT", "0.20"))
FEATURE_SHARE_LIMIT = float(os.getenv("DRIFT_FEATURE_SHARE", "0.5"))


def _rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(d, 0))


def mmd_test(x: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int = 0) -> tuple[float, float]:
    """Unbiased MMD^2 with RBF kernel (median heuristic) and permutation p-value."""
    rng = np.random.default_rng(seed)
    z = np.vstack([x, y])
    sample = z[rng.choice(len(z), min(len(z), 500), replace=False)]
    d = np.sqrt(np.maximum(((sample[:, None] - sample[None]) ** 2).sum(-1), 0))
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    k = _rbf(z, z, 1.0 / (2 * med ** 2))
    n = len(x)

    def stat(idx):
        a, b = idx[:n], idx[n:]
        kxx, kyy, kxy = k[np.ix_(a, a)], k[np.ix_(b, b)], k[np.ix_(a, b)]
        m, l = len(a), len(b)
        return ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                + (kyy.sum() - np.trace(kyy)) / (l * (l - 1)) - 2 * kxy.mean())

    idx = np.arange(len(z))
    observed = stat(idx)
    perms = [stat(rng.permutation(idx)) for _ in range(n_perm)]
    p = (1 + sum(v >= observed for v in perms)) / (n_perm + 1)
    return float(observed), float(p)


def ood_rate(ref: np.ndarray, cur: np.ndarray, q: float = 0.99) -> float:
    lw = LedoitWolf().fit(ref)
    thr = np.quantile(lw.mahalanobis(ref), q)
    return float(np.mean(lw.mahalanobis(cur) > thr))


def evidently_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame, out_html: Path) -> float:
    """Return the share of drifted columns; save the HTML report."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        log.warning("Evidently not installed - skipping HTML report")
        return 0.0
    snap = Report([DataDriftPreset()]).run(cur_df, ref_df)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(out_html))
    for m in snap.dict()["metrics"]:
        if m["metric_name"].startswith("DriftedColumnsCount"):
            return float(m["value"]["share"])
    return 0.0


def run_drift(ref_emb: np.ndarray, ref_margin: np.ndarray, cur_emb: np.ndarray,
              cur_margin: np.ndarray, report_dir: Path, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    if len(ref_emb) > 1000:
        keep = rng.choice(len(ref_emb), 1000, replace=False)
        ref_emb, ref_margin = ref_emb[keep], ref_margin[keep]
    mmd, p = mmd_test(ref_emb, cur_emb, seed=seed)
    ood = ood_rate(ref_emb, cur_emb)

    pca = PCA(n_components=min(5, ref_emb.shape[1], len(ref_emb) - 1), random_state=seed).fit(ref_emb)
    cols = [f"pc{i + 1}" for i in range(pca.n_components_)]
    ref_df = pd.DataFrame(pca.transform(ref_emb), columns=cols).assign(score=ref_margin)
    cur_df = pd.DataFrame(pca.transform(cur_emb), columns=cols).assign(score=cur_margin)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = report_dir / f"drift_{stamp}.html"
    share = evidently_report(ref_df, cur_df, html)
    if html.exists():
        (report_dir / "latest.html").write_bytes(html.read_bytes())

    detected = bool(p < MMD_ALPHA or ood > OOD_LIMIT or share >= FEATURE_SHARE_LIMIT)
    return {"mmd": mmd, "mmd_p_value": p, "ood_rate": ood, "drifted_features_share": share,
            "drift_detected": detected, "n_current": int(len(cur_emb)),
            "report_path": str(html) if html.exists() else ""}
