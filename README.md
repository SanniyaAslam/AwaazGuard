# AwaazGuard: Audio Deepfake & Voice-Clone Scam Detection

AwaazGuard detects AI-generated (spoofed) speech such as voice-clone scam calls and voice notes.
This repository contains **Deliverable 1** of the MLOps course project: a fully reproducible
ML pipeline covering data ingestion, validation, versioning, a baseline model, a main model,
evaluation on **unseen** attacks, MLflow experiment tracking and model registration.

> Deliverable 1 scope: no Docker, cloud deployment, REST API or monitoring yet. Those come
> in later deliverables.

| | |
|---|---|
| **GitHub** | `https://github.com/<github-username>/awaazguard` |
| **DagsHub (DVC remote + MLflow)** | `https://dagshub.com/<dagshub-username>/awaazguard` |
| **MLflow UI** | `https://dagshub.com/<dagshub-username>/awaazguard.mlflow` |
| **Team** | Fatima Habib (23L-0881), Sanniya Aslam (23L-0672) |

---

## 1. Problem and approach

Deepfake-audio detectors are trained on fakes from known generators and fail quietly when a
new generator appears. We train on **known attacks** (ASVspoof 2019 LA train set, attacks
A01-A06) and evaluate on **unseen attacks** (eval set, A07-A19). This simulates the attack drift
that later deliverables will monitor and react to.

| Model | Role | Input | Notes |
|---|---|---|---|
| **LFCC + GMM** | Baseline | 60-dim LFCC (static + delta + delta-delta) | Official ASVspoof baseline family: one GMM per class, log-likelihood ratio score |
| **AASIST-L** | Main | Raw 16 kHz waveform (~4 s) | Spectro-temporal graph attention network, ~85k parameters ([Jung et al., ICASSP 2022](https://github.com/clovaai/aasist), MIT licence) |

**Metrics:** Equal Error Rate (EER, lower is better), ROC-AUC and accuracy at the dev-set EER
threshold, plus **EER per unseen attack** (A07-A19). Best accuracy is not the objective.

## 2. Dataset

**ASVspoof 2019 Logical Access (LA)**, from [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)
(~7.1 GB, research licence). It contains bona fide speech (VCTK speakers) and spoofed speech from
TTS and voice-conversion systems, in speaker-disjoint train, dev and eval splits.

The `prepare` stage builds a reproducible, stratified subset (configurable in `params.yaml`):

| Split | Utterances | Bona fide share | Attacks |
|---|---|---|---|
| train | 2,000 | 30% | A01-A06 (known) |
| dev | 600 | 30% | A01-A06 (known) |
| eval | 1,950 | 30% | A07-A19 (13 **unseen** attacks, balanced) |

The subset (`data/processed/`: FLAC files + `metadata.csv`) is the **versioned dataset**, tracked by
DVC and pushed to the DagsHub remote. The raw 7 GB archive is public and re-downloadable, so it
is tracked in `dvc.lock` (hash only) but not pushed.

## 3. Pipeline

```
ingest → validate → prepare ─┬─ featurize → train_baseline ─┐
                             └─ train_main ─────────────────┴─ evaluate → register
```

| Stage | Script | What it does | Outputs |
|---|---|---|---|
| `ingest` | `src/stages/ingest.py` | Resumable download + extraction of LA.zip (or copy from `ingest.local_dir`); skips if already present | `data/raw/LA`, `reports/ingest.json` |
| `validate` | `src/stages/validate.py` | Schema (5 columns), nulls, duplicates, ID formats, label and attack-ID ranges per split, file existence, sample rate, duration range, silence, clipping, speaker-disjoint splits. **Fails the pipeline** if any check fails. | `reports/validation.json` |
| `prepare` | `src/stages/prepare.py` | Seeded stratified subset, copy audio, write metadata | `data/processed/`, `reports/data_summary.json` |
| `featurize` | `src/stages/featurize.py` | LFCC extraction (NumPy/SciPy, deterministic) | `data/features/` |
| `train_baseline` | `src/stages/train_baseline.py` | Fit bona fide and spoof GMMs, dev EER, MLflow run | `models/baseline/` |
| `train_main` | `src/stages/train_main.py` | Train AASIST-L, log loss / dev EER / LR **per epoch**, keep best epoch, log PyTorch model | `models/main/`, learning curves |
| `evaluate` | `src/stages/evaluate.py` | Score eval (unseen attacks) with both models, per-attack EER, ROC, MLflow evaluation run | `reports/metrics.json`, plots |
| `register` | `src/stages/register.py` | Register AASIST-L in the MLflow Model Registry and move it to **Staging** | `reports/registry.json` |

Every run is tagged in MLflow with the **Git commit** and the **DVC data hash** of `data/processed`,
so each result can be traced back to its exact code and data.

## 4. Setup

**Requirements:** Python 3.10-3.12, Git, about 16 GB free disk space (zip + extracted data), and an
NVIDIA GPU (recommended; a free Kaggle or Colab T4 works).

```bash
git clone https://github.com/<github-username>/awaazguard.git
cd awaazguard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Connect DagsHub (one time per machine)

1. Copy `.env.example` to `.env` and fill in the MLflow URI, your DagsHub username and an access token
   (DagsHub → User Settings → Tokens).
2. Configure DVC credentials locally. The `--local` flag means they are **never committed**:

```bash
dvc remote modify origin --local access_key_id <dagshub-access-token>
dvc remote modify origin --local secret_access_key <dagshub-access-token>
```

## 5. How to run

```bash
dvc pull          # optional: fetch the versioned subset + models from DagsHub
dvc repro         # runs every stage that is out of date (all of them on first run)
```

* First run downloads ~7.1 GB. If you already have the extracted `LA` folder, set
  `ingest.local_dir` in `params.yaml` to its path to skip the download.
* Change any value in `params.yaml` (e.g. `main.epochs`), run `dvc repro`, and DVC re-runs only
  the affected stages.
* Training on **CPU only**: set `main.batch_size: 8` and `main.epochs: 3`. AASIST needs about 0.4 GB
  RAM per sample in a batch on CPU.

Useful commands:

```bash
dvc dag                     # pipeline graph
dvc metrics show            # baseline vs main EER/AUC
dvc plots show              # HTML plots (per-attack EER, ROC, dev EER per epoch)
dvc push                    # upload data/processed + models to DagsHub
pytest -q                   # unit tests (EER, LFCC, AASIST forward pass)
```

### Smoke test without the 7 GB download

```bash
python scripts/make_mock_dataset.py --out mock_data/LA --per-split 150
dvc exp run -S ingest.local_dir=mock_data/LA -S main.epochs=1 -S main.max_batches_per_epoch=2 -S main.batch_size=4
```

Synthetic audio only checks that the plumbing works; its scores are meaningless.

## 6. Folder structure

```
awaazguard/
├── data/                     # DVC-managed, git-ignored
│   ├── raw/LA/               # ASVspoof 2019 LA (downloaded by `ingest`)
│   ├── processed/            # versioned subset: audio/<split>/*.flac + metadata.csv
│   └── features/             # LFCC features (.npz)
├── models/                   # DVC-managed, git-ignored
│   ├── baseline/             # gmm.pkl + meta.json (run id, dev threshold)
│   └── main/                 # aasist_best.pt + meta.json
├── reports/                  # small JSON/CSV/PNG outputs tracked in Git
├── scripts/
│   └── make_mock_dataset.py  # tiny synthetic dataset for smoke tests
├── src/
│   ├── features/lfcc.py      # LFCC front-end
│   ├── models/aasist.py      # AASIST (vendored, MIT)
│   ├── models/gmm.py         # LFCC-GMM baseline
│   ├── stages/               # one script per DVC stage
│   └── utils/                # params, paths, metrics (EER), MLflow setup, audio IO
├── tests/test_core.py
├── dvc.yaml                  # pipeline definition (8 named stages)
├── dvc.lock                  # exact hashes of every stage input/output
├── params.yaml               # all hyperparameters and settings
├── requirements.txt
├── .env.example              # MLflow / DagsHub credentials template
└── .gitignore                # no raw data, features, weights or secrets in Git
```

## 7. Reproducibility guarantees

* All randomness (subset selection, crops, GMM init, PyTorch) comes from `seed` in `params.yaml`;
  cuDNN deterministic mode is on.
* `dvc.lock` pins the hash of every input and output; `dvc repro` on a clean clone rebuilds
  the same data subset and models.
* No raw data, features or model weights are committed to Git (see `.gitignore`); they live in
  the DVC remote on DagsHub.
* MLflow runs record parameters, per-epoch metrics, artifacts (model, learning curves, per-attack
  EER, validation report, `params.yaml`), Git commit and data version.

## 8. References

* X. Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech", *Computer Speech & Language*, 2020.
* J. Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks", ICASSP 2022.
* M. Todisco et al., "Constant Q cepstral coefficients: A spoofing countermeasure for automatic speaker verification", *Computer Speech & Language*, 2017 (GMM baseline family).
