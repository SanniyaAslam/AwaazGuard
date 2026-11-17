"""Stage 2 - validate: schema, nulls, duplicates, label/attack ranges, file existence,
and audio-level checks (sample rate, duration range, silence, clipping).

Exits with a non-zero code if any check fails, which stops `dvc repro` before training.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.utils.audio import load_audio
from src.utils.common import (EXPECTED_ATTACKS, LABELS, REPORTS_DIR, SPLITS, audio_path,
                              get_logger, load_params, protocol_path, save_json)

log = get_logger("validate")
COLUMNS = ["speaker_id", "utt_id", "env", "attack_id", "label"]


def read_protocol(split: str) -> pd.DataFrame:
    df = pd.read_csv(protocol_path(split), sep=r"\s+", header=None, dtype=str,
                     engine="python")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{split}: expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df.columns = COLUMNS
    df["split"] = split
    return df


def check_split(df: pd.DataFrame, split: str, p: dict, seed: int) -> dict:
    res: dict = {"rows": int(len(df))}
    checks: dict[str, bool] = {}

    # --- schema / nulls / duplicates ---
    checks["no_nulls"] = bool(df[COLUMNS].notna().all().all())
    checks["no_duplicate_utt_ids"] = bool(~df["utt_id"].duplicated().any())
    checks["speaker_id_format"] = bool(df["speaker_id"].str.match(r"^LA_\d{4}$").all())
    checks["utt_id_format"] = bool(df["utt_id"].str.match(r"^LA_[TDE]_\d{7}$").all())

    # --- categorical ranges ---
    checks["labels_valid"] = bool(df["label"].isin(LABELS).all())
    bona = df["label"] == "bonafide"
    checks["bonafide_have_no_attack"] = bool((df.loc[bona, "attack_id"] == "-").all())
    spoof_attacks = set(df.loc[~bona, "attack_id"])
    checks["attack_ids_in_expected_range"] = spoof_attacks <= EXPECTED_ATTACKS[split]
    checks["both_classes_present"] = bool(bona.any() and (~bona).any())

    # --- file existence (all rows) ---
    missing = [u for u in df["utt_id"] if not audio_path(split, u).exists()]
    checks["all_audio_files_exist"] = len(missing) == 0
    res["missing_files"] = len(missing)

    # --- audio-level checks on a reproducible sample ---
    rng = np.random.default_rng(seed)
    n = min(p["audio_sample_per_split"], len(df))
    sample = df.iloc[rng.choice(len(df), size=n, replace=False)]
    sr_ok, dur_ok, silent, clipped, durations = 0, 0, 0, 0, []
    for u in sample["utt_id"]:
        path = audio_path(split, u)
        if not path.exists():
            continue
        wav, sr = load_audio(path)
        dur = len(wav) / sr
        durations.append(dur)
        sr_ok += sr == p["expected_sample_rate"]
        dur_ok += p["min_duration_s"] <= dur <= p["max_duration_s"]
        silent += float(np.sqrt(np.mean(wav ** 2))) < 1e-4
        clipped += float(np.mean(np.abs(wav) >= 0.999)) > 0.01
    checks["sample_rate_correct"] = sr_ok == n
    checks["duration_in_range"] = dur_ok == n
    checks["silence_below_limit"] = silent / max(n, 1) <= p["max_silent_fraction"]
    checks["clipping_below_limit"] = clipped / max(n, 1) <= p["max_clipped_fraction"]

    res.update({
        "bonafide": int(bona.sum()),
        "spoof": int((~bona).sum()),
        "speakers": int(df["speaker_id"].nunique()),
        "attacks": sorted(spoof_attacks),
        "audio_sampled": n,
        "duration_mean_s": float(np.mean(durations)) if durations else 0.0,
        "duration_min_s": float(np.min(durations)) if durations else 0.0,
        "duration_max_s": float(np.max(durations)) if durations else 0.0,
        "checks": {k: bool(v) for k, v in checks.items()},
    })
    return res


def main() -> None:
    params = load_params()
    p, seed = params["validate"], params["seed"]
    report, frames = {}, {}
    for split in SPLITS:
        frames[split] = read_protocol(split)
        report[split] = check_split(frames[split], split, p, seed)
        log.info("%s: %s", split, report[split]["checks"])

    # Cross-split check: official splits must be speaker-disjoint
    spk = {s: set(frames[s]["speaker_id"]) for s in SPLITS}
    disjoint = not (spk["train"] & spk["dev"] or spk["train"] & spk["eval"]
                    or spk["dev"] & spk["eval"])
    report["cross_split"] = {"checks": {"speaker_disjoint_splits": disjoint}}

    failed = [f"{s}.{k}" for s, r in report.items() for k, ok in r["checks"].items() if not ok]
    report["passed"] = not failed
    report["failed_checks"] = failed
    save_json(report, REPORTS_DIR / "validation.json")

    if failed:
        log.error("Validation FAILED: %s", failed)
        sys.exit(1)
    log.info("All validation checks passed.")


if __name__ == "__main__":
    main()
