"""Continuous red-team: generate fresh fakes, then gate a candidate model against the champion.

  # 1) build a new red-team batch (fakes + real speech, all passed through phone/opus/mp3/noise)
  python -m src.redteam.run_redteam generate --generators piper --n-fake 200 --n-real 200
  # 2) version it
  dvc add data/redteam && git add data/redteam.dvc && git commit -m "red-team batch"
  # 3) gate: candidate (Staging) vs champion (Production). Exit code 1 = blocked.
  python -m src.redteam.run_redteam gate
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.redteam import channel
from src.redteam.generators import build
from src.serving.model_store import load_from_registry
from src.utils.common import (METADATA_CSV, REPORTS_DIR, ROOT, get_logger, load_params,
                              save_json)
from src.utils.metrics import compute_eer

log = get_logger("redteam")

# Social-engineering style prompts (no real names, numbers or bank details)
PROMPTS = [
    "Hello, it's me. I lost my phone, this is a new number, please save it.",
    "Mom, I am in trouble, I need you to send some money right now, please don't tell anyone.",
    "This is your bank's security department, we noticed an unusual transaction on your account.",
    "Your parcel is held at customs, you need to pay a small clearance fee today.",
    "Hi, can you approve the payment I sent you? The manager needs it before noon.",
    "I'm calling from the telecom company, your SIM will be blocked unless you verify your details.",
    "Brother, my card isn't working, can you transfer something now, I will return it tonight.",
    "Congratulations, you have won a prize, we only need a verification code to proceed.",
    "The meeting moved to three o'clock, please forward the document to the new address.",
    "Please don't call back, just send it to the account in my message.",
    "Assalam o alaikum, I need a quick favour, it's urgent, please reply.",
    "We are updating our records, could you confirm the one time password you just received?",
]


def _real_pool(n: int, rng: np.random.Generator) -> list[Path]:
    """Bona fide clips from the versioned dataset (dev + eval, never used for training)."""
    meta = pd.read_csv(METADATA_CSV)
    real = meta[(meta["label"] == "bonafide") & meta["split"].isin(["dev", "eval"])]
    pick = real.sample(n=min(n, len(real)), random_state=int(rng.integers(1 << 31)))
    return [ROOT / p for p in pick["path"]]


def generate(args) -> None:
    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "audio").mkdir(parents=True)
    gens = build(args.generators.split(","), args.voices_dir)
    reals = _real_pool(args.n_real + args.n_fake, rng)
    rows = []

    for i in tqdm(range(args.n_fake), desc="fakes"):
        g = gens[i % len(gens)]
        ref = sf.read(reals[i % len(reals)], dtype="float32")[0] if g.name == "xtts" else None
        wav = g.generate(PROMPTS[int(rng.integers(len(PROMPTS)))], rng, reference_wav=ref)
        ch = channel.CHANNELS[int(rng.integers(len(channel.CHANNELS)))]
        wav = channel.apply(wav, ch, rng)
        f = f"audio/fake_{i:05d}.wav"
        sf.write(out / f, wav, channel.SR)
        rows.append({"file": f, "label": "spoof", "generator": getattr(g, "last_variant", g.name),
                     "channel": ch})

    for i, src in enumerate(tqdm(reals[: args.n_real], desc="real")):
        wav, _ = sf.read(src, dtype="float32")
        ch = channel.CHANNELS[int(rng.integers(len(channel.CHANNELS)))]
        f = f"audio/real_{i:05d}.wav"
        sf.write(out / f, channel.apply(wav, ch, rng), channel.SR)
        rows.append({"file": f, "label": "bonafide", "generator": "human", "channel": ch})

    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    save_json({"created": datetime.now().isoformat(timespec="seconds"),
               "generators": args.generators, "n_fake": args.n_fake, "n_real": len(reals[: args.n_real]),
               "seed": args.seed}, out / "batch_info.json")
    log.info("Red-team batch written to %s (%d clips)", out, len(rows))


def score_batch(model, batch_dir: Path) -> pd.DataFrame:
    man = pd.read_csv(batch_dir / "manifest.csv")
    margins = []
    for f in tqdm(man["file"], desc=f"score v{model.version}"):
        wav, _ = sf.read(batch_dir / f, dtype="float32")
        margins.append(model.predict(wav)["margin"])
    return man.assign(margin=margins)


def eer_of(df: pd.DataFrame) -> float:
    return compute_eer(df.loc[df.label == "bonafide", "margin"].to_numpy(),
                       df.loc[df.label == "spoof", "margin"].to_numpy())[0]


def gate(args) -> int:
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.utils.tracking import setup_mlflow

    params = load_params()
    rp = params["redteam"]
    setup_mlflow(params["mlflow"]["experiment_name"])
    name = params["register"]["model_name"]
    client = MlflowClient()
    batch = ROOT / (args.batch or rp["batch_dir"])

    cand = load_from_registry(name, "Staging")
    champ = load_from_registry(name, "Production")
    if cand is None:
        log.error("No candidate in Staging")
        return 1

    c_df = score_batch(cand, batch)
    result = {"batch": str(batch.relative_to(ROOT)), "candidate_version": cand.version,
              "candidate_redteam_eer": eer_of(c_df),
              "candidate_by_channel": {ch: eer_of(g) for ch, g in c_df.groupby("channel")
                                       if g.label.nunique() == 2}}
    tags_c = client.get_model_version(name, cand.version).tags
    reasons = []
    if champ is not None:
        h_df = score_batch(champ, batch)
        result.update({"champion_version": champ.version, "champion_redteam_eer": eer_of(h_df)})
        if result["candidate_redteam_eer"] > result["champion_redteam_eer"] + rp["max_eer_increase"]:
            reasons.append("worse on red-team fakes")
        tags_h = client.get_model_version(name, champ.version).tags
        if rp["require_unseen_improvement"] and "eval_eer" in tags_c and "eval_eer" in tags_h:
            result["candidate_eval_eer"] = float(tags_c["eval_eer"])
            result["champion_eval_eer"] = float(tags_h["eval_eer"])
            if result["candidate_eval_eer"] > result["champion_eval_eer"]:
                reasons.append("worse on unseen ASVspoof attacks")
    result["passed"] = not reasons
    result["reasons"] = reasons
    save_json(result, REPORTS_DIR / "redteam_gate.json")

    mv = client.get_model_version(name, cand.version)
    client.set_model_version_tag(name, cand.version, "redteam_gate", "pass" if not reasons else "fail")
    client.set_model_version_tag(name, cand.version, "redteam_eer", f"{result['candidate_redteam_eer']:.4f}")
    with mlflow.start_run(run_name=f"redteam-gate-v{cand.version}"):
        mlflow.set_tags({"model_type": "redteam_gate", "candidate_run_id": mv.run_id})
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, float)})
        mlflow.log_artifact(str(REPORTS_DIR / "redteam_gate.json"))

    if reasons:
        log.error("RED-TEAM GATE FAILED for v%s: %s", cand.version, "; ".join(reasons))
        if args.archive_on_fail:
            client.transition_model_version_stage(name, cand.version, "Archived")
        return 1
    log.info("Red-team gate PASSED for v%s (EER %.4f)", cand.version, result["candidate_redteam_eer"])
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--generators", default="piper", help="comma list: piper,xtts,stub")
    g.add_argument("--voices-dir", default="voices")
    g.add_argument("--n-fake", type=int, default=200)
    g.add_argument("--n-real", type=int, default=200)
    g.add_argument("--out", default="data/redteam/latest")
    g.add_argument("--seed", type=int, default=int(datetime.now().strftime("%Y%m%d")))
    t = sub.add_parser("gate")
    t.add_argument("--batch", default=None)
    t.add_argument("--archive-on-fail", action="store_true")
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a)
    else:
        sys.exit(gate(a))


if __name__ == "__main__":
    main()
