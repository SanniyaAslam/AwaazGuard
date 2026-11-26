"""Decision logic for the CI release gates.

Kept free of heavy imports (no torch, no MLflow) so it can be unit-tested in seconds and
reused by the CI scripts:

* ``validate_metadata``  - data validation on the versioned dataset (``data/processed``)
* ``decide_gate``        - champion vs challenger model comparison on a fixed test set
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import EXPECTED_ATTACKS, LABELS, SPLITS

REQUIRED_COLUMNS = ["utt_id", "speaker_id", "split", "attack_id", "label", "path", "duration_s"]
FEEDBACK_SPEAKER = "FEEDBACK"      # production clips added by `prepare` (train split only)


def validate_metadata(meta: pd.DataFrame, root: Path, p: dict, seed: int = 42) -> dict:
    """Schema, quality and leakage checks on the processed dataset.

    ``p`` is the ``validate`` section of params.yaml. Returns a report with a ``checks``
    dict (name -> bool), the list of ``failed_checks`` and an overall ``passed`` flag.
    """
    checks: dict[str, bool] = {}
    checks["required_columns_present"] = all(c in meta.columns for c in REQUIRED_COLUMNS)
    if not checks["required_columns_present"]:
        return _report(checks, {"rows": int(len(meta))})

    checks["no_nulls"] = bool(meta[REQUIRED_COLUMNS].notna().all().all())
    checks["no_duplicate_utt_ids"] = not bool(meta["utt_id"].duplicated().any())
    checks["splits_valid"] = bool(meta["split"].isin(SPLITS).all()) and set(meta["split"]) >= set(SPLITS)
    checks["labels_valid"] = bool(meta["label"].isin(LABELS).all())

    bona = meta["label"] == "bonafide"
    checks["bonafide_have_no_attack"] = bool((meta.loc[bona, "attack_id"] == "-").all())

    official = meta[meta["speaker_id"] != FEEDBACK_SPEAKER]
    feedback = meta[meta["speaker_id"] == FEEDBACK_SPEAKER]
    checks["attack_ids_in_expected_range"] = all(
        set(official.loc[(official["split"] == s) & (official["label"] == "spoof"), "attack_id"])
        <= EXPECTED_ATTACKS[s] for s in SPLITS)
    checks["feedback_only_in_train"] = bool((feedback["split"] == "train").all())
    checks["both_classes_in_every_split"] = all(
        set(meta.loc[meta["split"] == s, "label"]) == set(LABELS) for s in SPLITS)

    # leakage: official splits must be speaker-disjoint
    spk = {s: set(official.loc[official["split"] == s, "speaker_id"]) for s in SPLITS}
    checks["speaker_disjoint_splits"] = not (spk["train"] & spk["dev"] or spk["train"] & spk["eval"]
                                             or spk["dev"] & spk["eval"])

    # files
    missing = [x for x in meta["path"] if not (root / str(x)).exists()]
    checks["all_audio_files_exist"] = len(missing) == 0

    # audio-level checks on a reproducible sample per split
    import soundfile as sf

    rng = np.random.default_rng(seed)
    sr_ok = dur_ok = silent = clipped = n_checked = 0
    for s in SPLITS:
        sub = meta[(meta["split"] == s) & meta["path"].map(lambda x: (root / str(x)).exists())]
        if sub.empty:
            continue
        sub = sub.iloc[rng.choice(len(sub), size=min(p["audio_sample_per_split"], len(sub)), replace=False)]
        for x in sub["path"]:
            wav, sr = sf.read(root / str(x), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            dur = len(wav) / sr
            n_checked += 1
            sr_ok += sr == p["expected_sample_rate"]
            dur_ok += p["min_duration_s"] <= dur <= p["max_duration_s"]
            silent += float(np.sqrt(np.mean(wav ** 2))) < 1e-4
            clipped += float(np.mean(np.abs(wav) >= 0.999)) > 0.01
    n = max(n_checked, 1)
    checks["sample_rate_correct"] = sr_ok == n_checked
    checks["duration_in_range"] = dur_ok == n_checked
    checks["silence_below_limit"] = silent / n <= p["max_silent_fraction"]
    checks["clipping_below_limit"] = clipped / n <= p["max_clipped_fraction"]

    stats = {"rows": int(len(meta)), "missing_files": len(missing), "audio_checked": n_checked,
             "per_split": {s: int((meta["split"] == s).sum()) for s in SPLITS}}
    return _report(checks, stats)


def _report(checks: dict[str, bool], stats: dict) -> dict:
    failed = [k for k, ok in checks.items() if not ok]
    return {**stats, "checks": {k: bool(v) for k, v in checks.items()},
            "failed_checks": failed, "passed": not failed}


def decide_gate(champion_eer: float, candidate_eer: float | None, tolerance: float,
                max_eer: float) -> dict:
    """Release-gate decision. The pipeline is blocked (``passed`` False) when

    * the champion itself is worse than the absolute ceiling ``max_eer`` (a regression in the
      serving code, data or registry), or
    * a challenger exists and is worse than the champion by more than ``tolerance``.
    """
    reasons: list[str] = []
    if champion_eer > max_eer:
        reasons.append(f"champion EER {champion_eer:.4f} is above the ceiling {max_eer:.4f}")
    delta = None
    if candidate_eer is not None:
        delta = candidate_eer - champion_eer
        if delta > tolerance:
            reasons.append(f"challenger EER {candidate_eer:.4f} is worse than champion "
                           f"{champion_eer:.4f} by {delta:.4f} (allowed {tolerance:.4f})")
    return {"passed": not reasons, "reasons": reasons, "champion_eer": champion_eer,
            "candidate_eer": candidate_eer, "delta": delta, "tolerance": tolerance, "max_eer": max_eer,
            "challenger_present": candidate_eer is not None}
