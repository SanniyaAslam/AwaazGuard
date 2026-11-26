"""CI data validation: schema + quality + leakage checks on the versioned dataset.

  dvc pull data/processed
  python -m scripts.validate_processed

Exit code 1 (and a clear list of failed checks) if the dataset is corrupted, incomplete or leaky.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from src.utils.common import METADATA_CSV, REPORTS_DIR, ROOT, get_logger, load_params, save_json
from src.utils.gates import validate_metadata

log = get_logger("validate_processed")


def main() -> None:
    params = load_params()
    if not METADATA_CSV.exists():
        sys.exit(f"{METADATA_CSV} not found - run `dvc pull data/processed` first")
    report = validate_metadata(pd.read_csv(METADATA_CSV), ROOT, params["validate"], params["seed"])
    save_json(report, REPORTS_DIR / "validation_processed.json")

    lines = ["### Data validation (data/processed)", "", "| check | result |", "|---|---|"]
    lines += [f"| {k} | {'pass' if ok else '**FAIL**'} |" for k, ok in report["checks"].items()]
    print("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    if not report["passed"]:
        log.error("Data validation FAILED: %s", report["failed_checks"])
        sys.exit(1)
    log.info("Data validation passed (%d rows, %d audio files decoded).", report["rows"],
             report["audio_checked"])


if __name__ == "__main__":
    main()
