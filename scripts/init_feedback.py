"""Create an empty, DVC-trackable feedback folder (run once, then `dvc add data/feedback`)."""
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "data" / "feedback"
(root / "audio").mkdir(parents=True, exist_ok=True)
labels = root / "labels.csv"
if not labels.exists():
    labels.write_text("prediction_id,file,label,source\n")
(root / "audio" / ".keep").touch()
print(f"Feedback folder ready at {root}")
