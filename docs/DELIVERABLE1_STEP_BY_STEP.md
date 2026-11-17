# Deliverable 1: step-by-step guide (Fatima + Sanniya)

Follow the phases in order. **F** = Fatima, **S** = Sanniya. Every commit is made from the
person's own GitHub account. The course requires commits from all group members.

---

## Phase 0: Accounts (both, 10 min)

1. Both have GitHub accounts, and both sign in to **https://dagshub.com** using "Sign in with GitHub".
2. **F**: on GitHub create an **empty** public repo named `awaazguard` (no README or .gitignore,
   since we have our own). Then Settings → Collaborators → add Sanniya.
3. **S**: accept the GitHub invitation email.

## Phase 1: Repo skeleton (F)

Unzip `awaazguard_deliverable1.zip`. Work inside the `awaazguard/` folder.

```bash
cd awaazguard
git init -b main
git remote add origin https://github.com/<F-github>/awaazguard.git
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Commit F1: project skeleton**
```bash
git add .gitignore requirements.txt .env.example params.yaml README.md docs/ reports/.gitkeep \
        src/__init__.py src/utils/ src/stages/__init__.py src/features/__init__.py src/models/__init__.py
git commit -m "Project skeleton: config, utils, README, gitignore"
```

**Commit F2: data ingestion and validation**
```bash
git add src/stages/ingest.py src/stages/validate.py src/stages/prepare.py
git commit -m "Data pipeline: ingest ASVspoof 2019 LA, validation checks, stratified subset"
git push -u origin main
```

## Phase 2: DagsHub + DVC remote (F)

1. DagsHub → **Create** → **New Repository** → **Connect a repository** → GitHub → choose `awaazguard`.
   DagsHub now mirrors the GitHub repo and gives you a free DVC remote **and** an MLflow server.
2. DagsHub repo → **Settings → Collaborators** → add Sanniya (Write access).
3. DagsHub repo → green **Remote** button → **Data → DVC** tab. Copy the commands shown there.
   They look like this:

```bash
dvc init
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<F-dagshub>/awaazguard.s3
dvc remote default origin
# credentials: LOCAL ONLY, never committed
dvc remote modify origin --local access_key_id <F-token>
dvc remote modify origin --local secret_access_key <F-token>
```

4. **Remote → Experiments** tab: copy the MLflow tracking URI
   (`https://dagshub.com/<F-dagshub>/awaazguard.mlflow`). Then:

```bash
cp .env.example .env      # fill URI, DagsHub username, token (User Settings → Tokens)
```

**Commit F3: DVC init + remote**
```bash
git add .dvc/config .dvc/.gitignore .dvcignore
git commit -m "Initialise DVC with DagsHub S3 remote"
git push
```

## Phase 3: Models (S)

```bash
git clone https://github.com/<F-github>/awaazguard.git && cd awaazguard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Copy these files from the zip into the cloned folder (same paths), then commit:

**Commit S1: LFCC features + GMM baseline**
```bash
git add src/features/lfcc.py src/stages/featurize.py src/models/gmm.py src/stages/train_baseline.py
git commit -m "Baseline: LFCC front-end and LFCC-GMM model with MLflow logging"
```

**Commit S2: AASIST-L main model**
```bash
git add src/models/aasist.py src/models/AASIST_LICENSE.txt src/stages/train_main.py
git commit -m "Main model: AASIST-L training with per-epoch MLflow logging"
```

**Commit S3: evaluation, registry, tests**
```bash
git add src/stages/evaluate.py src/stages/register.py tests/ scripts/
git commit -m "Evaluation on unseen attacks, MLflow model registration, unit tests"
git push
```

S also sets up her own credentials (needed only if she runs the pipeline):
`.env` with **her** DagsHub username + token, and the two `dvc remote modify origin --local ...`
lines with her token.

## Phase 4: Pipeline definition (F)

```bash
git pull
git add dvc.yaml
git commit -m "DVC pipeline: 8 named stages from ingest to register"
git push
```

## Phase 5: Run the pipeline (whoever has the GPU)

**Option A: own machine with an NVIDIA GPU**
```bash
pytest -q          # 6 tests should pass
dvc repro          # first run downloads ~7.1 GB, then trains both models
```

**Option B: Kaggle Notebook (free T4 GPU, internet ON)**
```python
!git clone https://github.com/<F-github>/awaazguard.git
%cd awaazguard
!pip install -q -r requirements.txt
import os
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/<F-dagshub>/awaazguard.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "<dagshub-username>"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<token>"           # use Kaggle Secrets, not plain text
!dvc remote modify origin --local access_key_id <token>
!dvc remote modify origin --local secret_access_key <token>
!git config user.name "<name>" && git config user.email "<github email>"
!dvc repro
!dvc push
```
To push the `dvc.lock`/`reports` commit from Kaggle, use a GitHub personal access token:
`!git push https://<github-user>:<github-PAT>@github.com/<F-github>/awaazguard.git main`.
Alternatively, download `dvc.lock` and `reports/` and commit them from your laptop.

**If the run is slow on CPU:** `params.yaml` → `main.batch_size: 8`, `main.epochs: 3`.

After it finishes:
```bash
dvc push                                   # data/processed + models -> DagsHub
git add dvc.lock reports/
git commit -m "Pipeline run: data version, metrics, plots, registry record"
git push
```

## Phase 6: A second experiment run (either person, for the screenshots)

The PDF needs at least 2 MLflow runs. You already have 3 (baseline, main, evaluation). For a
clearer comparison, add a hyperparameter variant:

```bash
dvc exp run -S main.lr=0.0003 -S main.epochs=6
```
This logs another `main-aasist-l` run to MLflow with different parameters. Then:
`dvc exp show` lists both. Keep the original as the committed result.

## Phase 7: Screenshots for the PDF

| # | Where | What to capture |
|---|---|---|
| 1 | DagsHub → Experiments (or `.mlflow` UI) | Runs table showing **baseline-lfcc-gmm**, **main-aasist-l**, **evaluation-unseen-attacks** with metrics columns |
| 2 | Click **main-aasist-l** → Metrics | `dev_eer` / `train_loss` charts over epochs |
| 3 | MLflow UI → **Models** → `awaazguard-aasist` | Version 1 with stage **Staging** |
| 4 | Terminal: `dvc dag` | ASCII pipeline graph (or DagsHub repo page → the pipeline graph view) |
| 5 (optional) | DagsHub repo → Data | `data/processed` tracked by DVC |
| 6 (optional) | `reports/eval_overview.png` | Per-attack EER + ROC figure |

Paste them into `docs/Deliverable1_Report_AwaazGuard.docx` where the placeholders are, fill
in the links and the EER numbers from `dvc metrics show`, then **File → Export → PDF**.

## Checklist before submitting

- [ ] GitHub has commits from **both** Fatima and Sanniya
- [ ] No `data/`, `models/`, `.env`, `.pth`, `.flac` files on GitHub
- [ ] `dvc push` done: DagsHub shows `data/processed` and `models/` as DVC-tracked
- [ ] MLflow on DagsHub (not local) shows at least 2 runs with params, metrics and artifacts
- [ ] `awaazguard-aasist` in the Model Registry has stage **Staging**
- [ ] `dvc status` says "Data and pipelines are up to date"
- [ ] README links replaced with your real GitHub/DagsHub usernames
- [ ] PDF is 3-4 pages with links, screenshots and write-up
