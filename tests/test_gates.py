"""Tests for the CI gates: data validation catches bad data, the model gate blocks regressions."""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.utils.gates import decide_gate, validate_metadata

P = {"expected_sample_rate": 16000, "min_duration_s": 0.5, "max_duration_s": 15.0,
     "audio_sample_per_split": 50, "max_silent_fraction": 0.02, "max_clipped_fraction": 0.02}


def _wav(path, sr=16000, sec=1.5, amp=0.1, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = (amp * np.random.default_rng(seed).standard_normal(int(sr * sec))).astype(np.float32)
    sf.write(path, x, sr)


@pytest.fixture()
def dataset(tmp_path):
    """Tiny but valid processed dataset: 3 splits, 2 bona fide + 2 spoof each, distinct speakers."""
    rows, seed = [], 0
    attacks = {"train": ["A01", "A02"], "dev": ["A03", "A04"], "eval": ["A07", "A08"]}
    for si, split in enumerate(["train", "dev", "eval"]):
        for k in range(4):
            bona = k < 2
            uid = f"LA_{split[0].upper()}_{si}{k:06d}"
            rel = f"data/processed/audio/{split}/{uid}.flac"
            _wav(tmp_path / rel, seed=seed)
            seed += 1
            rows.append({"utt_id": uid, "speaker_id": f"LA_{si}{k:03d}", "split": split,
                         "attack_id": "-" if bona else attacks[split][k - 2],
                         "label": "bonafide" if bona else "spoof", "path": rel, "duration_s": 1.5})
    return pd.DataFrame(rows), tmp_path


def test_valid_dataset_passes(dataset):
    meta, root = dataset
    rep = validate_metadata(meta, root, P)
    assert rep["passed"], rep["failed_checks"]


def test_missing_file_is_caught(dataset):
    meta, root = dataset
    (root / meta.loc[3, "path"]).unlink()
    rep = validate_metadata(meta, root, P)
    assert "all_audio_files_exist" in rep["failed_checks"]


def test_null_label_duplicate_and_bad_label_are_caught(dataset):
    meta, root = dataset
    bad = meta.copy()
    bad.loc[0, "label"] = None
    assert "no_nulls" in validate_metadata(bad, root, P)["failed_checks"]
    bad = meta.copy()
    bad.loc[1, "utt_id"] = bad.loc[0, "utt_id"]
    assert "no_duplicate_utt_ids" in validate_metadata(bad, root, P)["failed_checks"]
    bad = meta.copy()
    bad.loc[0, "label"] = "maybe"
    assert "labels_valid" in validate_metadata(bad, root, P)["failed_checks"]


def test_speaker_leakage_is_caught(dataset):
    meta, root = dataset
    bad = meta.copy()
    bad.loc[bad["split"] == "eval", "speaker_id"] = bad.loc[bad["split"] == "train", "speaker_id"].values
    assert "speaker_disjoint_splits" in validate_metadata(bad, root, P)["failed_checks"]


def test_corrupted_audio_is_caught(dataset):
    meta, root = dataset
    _wav(root / meta.loc[0, "path"], sr=8000)                 # wrong sample rate
    assert "sample_rate_correct" in validate_metadata(meta, root, P)["failed_checks"]
    meta2, root2 = dataset
    _wav(root2 / meta2.loc[1, "path"], sr=16000, amp=0.0)     # silent file
    assert "silence_below_limit" in validate_metadata(meta2, root2, P)["failed_checks"]


def test_unseen_attack_in_train_is_caught(dataset):
    meta, root = dataset
    bad = meta.copy()
    bad.loc[(bad["split"] == "train") & (bad["label"] == "spoof"), "attack_id"] = "A10"
    assert "attack_ids_in_expected_range" in validate_metadata(bad, root, P)["failed_checks"]


def test_gate_passes_better_or_equal_challenger():
    assert decide_gate(0.20, 0.18, 0.01, 0.35)["passed"]
    assert decide_gate(0.20, 0.205, 0.01, 0.35)["passed"]        # within tolerance
    assert decide_gate(0.20, None, 0.01, 0.35)["passed"]         # no challenger: champion check only


def test_gate_blocks_worse_challenger_or_bad_champion():
    r = decide_gate(0.20, 0.25, 0.01, 0.35)
    assert not r["passed"] and "worse than champion" in r["reasons"][0]
    assert not decide_gate(0.40, None, 0.01, 0.35)["passed"]     # champion above the ceiling
