"""Stage 1 - ingest: download + extract ASVspoof 2019 LA (or reuse a local copy).

Idempotent: if data/raw/LA already holds the protocols and audio folders, nothing is
re-downloaded, so `dvc repro` is cheap after the first run.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from src.utils.common import (ROOT, SPLITS, get_logger, la_root, load_params,
                              protocol_path, save_json, REPORTS_DIR)

log = get_logger("ingest")


def is_complete(root: Path) -> bool:
    return all(protocol_path(s).exists() for s in SPLITS) and all(
        (root / f"ASVspoof2019_LA_{s}" / "flac").is_dir() for s in SPLITS
    )


def download(url: str, dest: Path) -> None:
    """Resumable HTTP download with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "awaazguard-ingest/1.0"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(req) as resp:
        if existing and resp.status != 206:  # server ignored Range -> restart
            existing = 0
        total = int(resp.headers.get("Content-Length", 0)) + existing
        mode = "ab" if existing else "wb"
        with open(dest, mode) as f, tqdm(total=total, initial=existing, unit="B",
                                         unit_scale=True, desc="LA.zip") as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def extract(zip_path: Path, out_dir: Path) -> None:
    log.info("Extracting %s -> %s", zip_path, out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.infolist(), desc="extract"):
            zf.extract(member, out_dir)


def main() -> None:
    p = load_params()["ingest"]
    target = la_root()

    if is_complete(target):
        log.info("ASVspoof 2019 LA already present at %s - skipping download.", target)
    elif p.get("local_dir"):
        src = Path(p["local_dir"]).expanduser().resolve()
        if not is_complete_dir(src):
            sys.exit(f"local_dir {src} does not look like an extracted LA folder")
        log.info("Copying local dataset from %s", src)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)
    else:
        zip_path = ROOT / p["zip_path"]
        if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
            log.info("Downloading ASVspoof 2019 LA from %s", p["url"])
            download(p["url"], zip_path)
        extract(zip_path, ROOT / p["extract_dir"])
        if p.get("delete_zip_after_extract"):
            zip_path.unlink(missing_ok=True)

    if not is_complete(target):
        sys.exit("Ingest failed: expected protocol files / flac folders are missing.")

    counts = {s: len(list((target / f"ASVspoof2019_LA_{s}" / "flac").glob("*.flac")))
              for s in SPLITS}
    log.info("Audio files per split: %s", counts)
    save_json({"source": p["url"] if not p.get("local_dir") else p["local_dir"],
               "flac_files": counts}, REPORTS_DIR / "ingest.json")


def is_complete_dir(src: Path) -> bool:
    return (src / "ASVspoof2019_LA_cm_protocols").is_dir() and all(
        (src / f"ASVspoof2019_LA_{s}" / "flac").is_dir() for s in SPLITS)


if __name__ == "__main__":
    main()
