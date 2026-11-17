"""Anti-spoofing metrics. Convention: higher score = more likely BONA FIDE."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and the threshold where miss rate == false-alarm rate."""
    bonafide_scores = np.asarray(bonafide_scores, dtype=np.float64)
    spoof_scores = np.asarray(spoof_scores, dtype=np.float64)
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return float("nan"), float("nan")

    scores = np.concatenate([bonafide_scores, spoof_scores])
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    sorted_scores = scores[order]

    # Sweep threshold over every score
    tar_cum = np.cumsum(labels)
    non_cum = len(spoof_scores) - (np.arange(1, len(labels) + 1) - tar_cum)
    frr = np.concatenate([[0.0], tar_cum / len(bonafide_scores)])      # bona fide rejected
    far = np.concatenate([[1.0], non_cum / len(spoof_scores)])         # spoof accepted
    thresholds = np.concatenate([[sorted_scores[0] - 1e-3], sorted_scores])

    idx = int(np.argmin(np.abs(frr - far)))
    eer = float((frr[idx] + far[idx]) / 2.0)
    return eer, float(thresholds[idx])


def summarize(scores: np.ndarray, labels: np.ndarray, attacks: np.ndarray,
              threshold: float | None = None) -> dict:
    """Overall EER/AUC/accuracy plus per-attack EER (each attack vs all bona fide)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    attacks = np.asarray(attacks)
    is_bona = labels == "bonafide"

    eer, eer_thr = compute_eer(scores[is_bona], scores[~is_bona])
    thr = eer_thr if threshold is None else threshold
    pred_bona = scores > thr
    acc = float(np.mean(pred_bona == is_bona))
    try:
        auc = float(roc_auc_score(is_bona.astype(int), scores))
    except ValueError:
        auc = float("nan")

    per_attack = {}
    for atk in sorted(set(attacks[~is_bona])):
        m = attacks == atk
        per_attack[atk] = compute_eer(scores[is_bona], scores[m])[0]

    return {
        "eer": eer,
        "eer_threshold": eer_thr,
        "threshold_used": thr,
        "accuracy": acc,
        "auc": auc,
        "n_bonafide": int(is_bona.sum()),
        "n_spoof": int((~is_bona).sum()),
        "per_attack_eer": per_attack,
    }
