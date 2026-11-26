"""Pull labelled production clips from the gateway into data/feedback (then `dvc add` it).

  python scripts/export_feedback.py --gateway http://<server-address> --token <ADMIN_TOKEN>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def export(gateway: str, token: str, out_dir: Path, all_items: bool = False) -> int:
    h = {"X-Admin-Token": token}
    with httpx.Client(base_url=gateway, headers=h, timeout=60) as c:
        items = c.get("/api/admin/feedback-export", params={"only_new": not all_items}).raise_for_status().json()
        (out_dir / "audio").mkdir(parents=True, exist_ok=True)
        labels_csv = out_dir / "labels.csv"
        old = pd.read_csv(labels_csv) if labels_csv.exists() else pd.DataFrame(
            columns=["prediction_id", "file", "label", "source"])
        new_rows = []
        for it in items:
            f = f"audio/{it['prediction_id']}.wav"
            (out_dir / f).write_bytes(c.get(f"/api/admin/audio/{it['prediction_id']}").raise_for_status().content)
            new_rows.append({"prediction_id": it["prediction_id"], "file": f,
                             "label": it["label"], "source": it["source"]})
        df = pd.concat([old, pd.DataFrame(new_rows)]).drop_duplicates("prediction_id", keep="last")
        df.to_csv(labels_csv, index=False)
        if items:
            c.post("/api/admin/feedback-export/ack", json=[i["feedback_id"] for i in items]).raise_for_status()
    print(f"Exported {len(items)} new labelled clips -> {out_dir} (total {len(df)})")
    return len(items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default=os.getenv("GATEWAY_URL", "http://localhost"))
    ap.add_argument("--token", default=os.getenv("ADMIN_TOKEN", ""))
    ap.add_argument("--out", default=str(ROOT / "data" / "feedback"))
    ap.add_argument("--all", action="store_true", help="re-export already exported items too")
    a = ap.parse_args()
    export(a.gateway, a.token, Path(a.out), a.all)


if __name__ == "__main__":
    main()
