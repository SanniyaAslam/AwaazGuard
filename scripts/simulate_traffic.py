"""Simulate production traffic (and a human annotator) against the gateway.

Sources:
  --source asvspoof            clips from data/processed (split from --split, default eval)
  --source itw --itw-dir DIR   In-the-Wild dataset folder containing meta.csv + wav files
                               (real-world shift -> should trigger the drift monitor)
  --source folder --dir DIR    any folder with labels.csv (file,label); a red-team batch folder works
                               too (its manifest.csv has the same two columns)

  python scripts/simulate_traffic.py --gateway http://<server-address> --source itw \
      --itw-dir ~/datasets/release_in_the_wild --n 300 --feedback-rate 0.5
"""
from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_items(a) -> list[tuple[Path, str]]:
    if a.source == "asvspoof":
        meta = pd.read_csv(ROOT / "data" / "processed" / "metadata.csv")
        meta = meta[meta["split"] == a.split]
        return [(ROOT / p, lab) for p, lab in zip(meta["path"], meta["label"])]
    if a.source == "itw":
        d = Path(a.itw_dir).expanduser()
        meta = pd.read_csv(d / "meta.csv")
        lab = meta["label"].str.lower().map(lambda s: "bonafide" if "bona" in s else "spoof")
        return [(d / f, l) for f, l in zip(meta["file"], lab)]
    d = Path(a.dir).expanduser()
    table = d / "labels.csv" if (d / "labels.csv").exists() else d / "manifest.csv"
    meta = pd.read_csv(table)
    return [(d / f, l) for f, l in zip(meta["file"], meta["label"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default=os.getenv("GATEWAY_URL", "http://localhost"))
    ap.add_argument("--source", choices=["asvspoof", "itw", "folder"], default="asvspoof")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--itw-dir")
    ap.add_argument("--dir")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--feedback-rate", type=float, default=0.5,
                    help="share of requests for which the true label is sent back")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    items = load_items(a)
    random.Random(a.seed).shuffle(items)
    items = items[: a.n]
    rng = random.Random(a.seed + 1)
    stats = {"ok": 0, "fail": 0, "feedback": 0, "correct": 0}

    def send(item):
        path, label = item
        with httpx.Client(base_url=a.gateway, timeout=120) as c:
            r = c.post("/api/predict", files={"file": (path.name, path.read_bytes())})
            if r.status_code != 200:
                stats["fail"] += 1
                return
            res = r.json()
            stats["ok"] += 1
            stats["correct"] += int((res["prob_fake"] >= 0.5) == (label == "spoof"))
            if rng.random() < a.feedback_rate:
                c.post("/api/feedback", json={"prediction_id": res["id"], "label": label,
                                              "source": "simulated_annotator"})
                stats["feedback"] += 1

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(send, items))
    acc = stats["correct"] / max(stats["ok"], 1)
    print(f"Sent {stats['ok']} ok / {stats['fail']} failed in {time.time() - t0:.0f}s | "
          f"feedback {stats['feedback']} | accuracy@0.5 {acc:.3f}")


if __name__ == "__main__":
    main()
