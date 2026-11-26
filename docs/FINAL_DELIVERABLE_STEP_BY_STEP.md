# Final deliverable: step-by-step guide (Sanniya + Fatima)

**S** = Sanniya, **F** = Fatima. Everything is free. You are on **Windows + PowerShell + VS Code**,
so every command below is PowerShell. Your repo is `https://github.com/SanniyaAslam/AwaazGuard`
(owned by Sanniya, Fatima is a collaborator), and DagsHub is also Sanniya's.

You already finished Deliverable 1 (pipeline, DVC, MLflow). This deliverable puts the model into a
**live, monitored, self-improving system**.

---

## 0. What the TA asks for, and where each item is covered

| TA requirement | Where it is covered | Phase |
|---|---|---|
| Containerized inference service + UI with live input and explanations | FastAPI gateway + Next.js site in one Docker image. The result page shows a **confidence score** (percentage), a **per-second risk band** (our saliency map: which seconds look machine-made), a waveform and a spectrogram. Deployed on an Azure VM. | C, D |
| Eval-gated CI/CD on every PR: unit tests, data validation, model evaluation gate; block if worse than production on a fixed test set | `.github/workflows/ci.yml`: `unit-tests`, `data-validation` (`scripts/validate_processed.py`) and `model-gate` (`scripts/model_gate.py`: Staging challenger vs Production champion on a fixed seeded test set). `cd.yml` deploys **only after CI succeeds**. Branch protection stops the merge. | E |
| Live observability: latency, errors, input/prediction drift | Prometheus + Grafana dashboard (latency p95, errors, verdicts, canary share) and a drift monitor (MMD, out-of-distribution rate, Evidently report) | F |
| Closed-loop retraining and rollback | Prefect `retrain_flow` (drift, then feedback + red-team data, then `dvc repro`, then red-team gate, then canary) and `canary_flow` (promote or roll back, with a reason) | G |
| Submit: GitHub repo, live URL, 5-minute demo video | | H, I |

## 1. The plan in one list

| Phase | What | Who |
|---|---|---|
| A | Add the new code to your repo with 14 commits (7 each) | both |
| B | Train the new model on Kaggle, register it, make it the champion | F (or S) |
| C | Run the whole system on your laptop with Docker | S |
| D | Create the Azure server with Terraform and deploy | F |
| E | CI/CD secrets, branch protection, prove the gate blocks a bad model | S |
| F | Look at Grafana and the drift monitor | both |
| G | Simulate drift, retrain, canary, promote or roll back | F + S |
| H | Record the 5-minute video | both |
| I | Submit | S |

Suggested days: **Day 1** A + B (start training, it takes hours), **Day 2** C + D, **Day 3** E + F,
**Day 4** G, **Day 5** rehearse and record H.

Three rules from the handbook that matter most: **never commit secrets** (`.env`, `deploy.env`,
`terraform.tfvars`, SSH keys, tokens), **never commit data or models** (DVC does that), and
**change settings in `params.yaml`**, not in code.

---

## Phase A: add the new code (14 commits, both of you)

The new zip (`awaazguard_project.zip`, or the updated one I gave you) contains the old Deliverable 1
files plus the new ones. Do **not** unzip it inside your repo. That caused the "untracked files would
be overwritten" error last time.

**Both of you, once:**
1. Unzip the new zip into a folder **outside** your repo, for example `C:\Users\<you>\Desktop\mlopsProject\final_zip`.
2. Open your repo folder in VS Code, open the terminal, make sure `(.venv)` shows.
3. Paste this helper (change the `$Z` path to the `awaazguard` folder inside your unzipped copy):

```powershell
$Z = "C:\Users\Sanni\Desktop\mlopsProject\final_zip\awaazguard"
function Cp($rel) { $dst = Join-Path (Get-Location) (Split-Path $rel -Parent); New-Item -ItemType Directory -Force $dst | Out-Null; Copy-Item (Join-Path $Z $rel) $dst -Recurse -Force }
```

`Cp "src\serving"` now copies that file or folder from the zip into the same place in your repo.
If the helper gives an error, copy the files by hand in File Explorer instead. You must paste the
helper again in every new terminal.

**Rules for every commit:** run `git pull` first. Copy only the files listed for your commit. Run
`git status` before committing and check that you see only those files. Never copy `docs\Deliverable1_Report_AwaazGuard.docx`.

Go in this order. S, F, S, F... Each person works from their own clone, and each commit is one block.

### 1. Sanniya: main model with channel augmentation and ONNX export
```powershell
git pull
Cp "src\stages\train_main.py"; Cp "src\stages\export_onnx.py"; Cp "src\stages\register.py"; Cp "src\redteam\__init__.py"; Cp "src\redteam\channel.py"
git add src/stages/train_main.py src/stages/export_onnx.py src/stages/register.py src/redteam/__init__.py src/redteam/channel.py
git commit -m "Main model: channel augmentation and ONNX export"
git push
```

### 2. Fatima: production feedback in the data pipeline
Before copying, open your current `params.yaml` and note any values you changed (for example
`main.epochs`, `main.batch_size`). Copy the new file, then set those values again.
```powershell
git pull
Cp "src\stages\prepare.py"; Cp "scripts\__init__.py"; Cp "scripts\init_feedback.py"; Cp "params.yaml"; Cp "dvc.yaml"
python scripts/init_feedback.py
dvc add data/feedback
dvc push
git add src/stages/prepare.py scripts/__init__.py scripts/init_feedback.py params.yaml dvc.yaml data/feedback.dvc data/.gitignore
git commit -m "Data pipeline: production feedback in training data, ONNX stage in DVC"
git push
```
If `git add` says `data/.gitignore` did not match any file, leave that name out.

### 3. Sanniya: model loading from the MLflow registry
```powershell
git pull
Cp "src\serving\__init__.py"; Cp "src\serving\audio.py"; Cp "src\serving\model_store.py"; Cp "src\serving\backends.py"; Cp "src\serving\inference_app.py"; Cp "requirements-serve.txt"
git add src/serving/__init__.py src/serving/audio.py src/serving/model_store.py src/serving/backends.py src/serving/inference_app.py requirements-serve.txt
git commit -m "Serving: load champion and candidate models from the MLflow registry"
git push
```

### 4. Fatima: gateway with canary routing, feedback and admin API
```powershell
git pull
Cp "src\serving\db.py"; Cp "src\serving\storage.py"; Cp "src\serving\gateway_app.py"; Cp "scripts\promote.py"; Cp "scripts\export_feedback.py"
git add src/serving/db.py src/serving/storage.py src/serving/gateway_app.py scripts/promote.py scripts/export_feedback.py
git commit -m "Gateway: website, canary routing, prediction log, feedback, admin API"
git push
```

### 5. Sanniya: web app
```powershell
git pull
Cp "frontend"
git add frontend
git commit -m "Web app: voice check with per-second explanation and operations dashboard"
git push
```
Check `git status` first. `frontend\node_modules` must not appear. If it does, stop and tell me.

### 6. Fatima: Docker image, compose stack, Prometheus and Grafana
```powershell
git pull
Cp "docker"; Cp "docker-compose.yml"; Cp ".dockerignore"; Cp "deploy.env.example"; Cp ".gitignore"
git add docker docker-compose.yml .dockerignore deploy.env.example .gitignore
git commit -m "Docker image, compose stack, Prometheus and Grafana"
git push
```

### 7. Sanniya: drift monitoring
```powershell
git pull
Cp "src\monitoring"
git add src/monitoring
git commit -m "Drift monitoring: MMD, OOD rate, Evidently"
git push
```

### 8. Fatima: Azure server with Terraform
```powershell
git pull
Cp "infra"
git add infra
git commit -m "Free Azure server with Terraform"
git push
```
`terraform.tfvars` is ignored by Git, so it never appears here.

### 9. Sanniya: eval-gated CI (data validation + model gate + tests)
Before you push this commit, add the three DagsHub secrets (`MLFLOW_TRACKING_URI`, `DAGSHUB_USERNAME`,
`DAGSHUB_TOKEN`, see the table in Phase E step 1) and finish Phase B, so a model is in Production. Otherwise the new CI
run is red, which is harmless but confusing.
```powershell
git pull
Cp "src\utils\gates.py"; Cp "scripts\validate_processed.py"; Cp "scripts\model_gate.py"; Cp "scripts\eval_registry_model.py"; Cp "tests\test_system.py"; Cp "tests\test_gates.py"; Cp ".github\workflows\ci.yml"
git add src/utils/gates.py scripts/validate_processed.py scripts/model_gate.py scripts/eval_registry_model.py tests/test_system.py tests/test_gates.py .github/workflows/ci.yml
git commit -m "Eval-gated CI: unit tests, data validation, champion vs challenger gate"
git push
```

### 10. Fatima: CD that deploys only after CI passes
```powershell
git pull
Cp ".github\workflows\cd.yml"
git add .github/workflows/cd.yml
git commit -m "CD: build, push to GHCR and deploy only after CI succeeds"
git push
```

### 11. Sanniya: red-team generators and release gate
```powershell
git pull
Cp "src\redteam\generators.py"; Cp "src\redteam\run_redteam.py"
git add src/redteam/generators.py src/redteam/run_redteam.py
git commit -m "Continuous red-team generators and release gate"
git push
```

### 12. Fatima: Prefect retrain flow and traffic simulator
```powershell
git pull
Cp "flows\__init__.py"; Cp "flows\retrain_flow.py"; Cp "scripts\simulate_traffic.py"; Cp "requirements-flows.txt"
git add flows/__init__.py flows/retrain_flow.py scripts/simulate_traffic.py requirements-flows.txt
git commit -m "Prefect retraining flow and traffic simulator"
git push
```

### 13. Sanniya: Prefect canary promote/rollback flow
```powershell
git pull
Cp "flows\canary_flow.py"
git add flows/canary_flow.py
git commit -m "Prefect canary promote/rollback flow"
git push
```

### 14. Fatima: README and docs
```powershell
git pull
Cp "README.md"; Cp "docs\FINAL_DELIVERABLE_STEP_BY_STEP.md"; Cp "docs\AwaazGuard_Handbook.pdf"
git add README.md docs/FINAL_DELIVERABLE_STEP_BY_STEP.md docs/AwaazGuard_Handbook.pdf
git commit -m "README and docs for the final submission"
git push
```

**Check it worked:** run `git status` (it should say "working tree clean"), run `pytest -q` (all tests
pass, it takes a minute the first time), and on GitHub open **Insights → Contributors**: both names
appear. If you mix up an order, nothing breaks. The final state of the repo is what counts.

---

## Phase B: train the new model and make it the champion (F, or S)

Your Deliverable 1 model has no ONNX files (`serving/`), so the app cannot load it. You need a new
registered version.

1. Kaggle → New Notebook → Accelerator **GPU T4 x2**, **Internet On**.
2. Add-ons → Secrets: `DAGSHUB_USER`, `DAGSHUB_TOKEN`, `GITHUB_PAT` (GitHub → Settings → Developer
   settings → Fine-grained tokens, repository `AwaazGuard`, **Contents: Read and write**). Tick each one
   for this notebook.
3. Run these cells in order:

```python
# Cell 1
from kaggle_secrets import UserSecretsClient
import os
s = UserSecretsClient()
for k in ["DAGSHUB_USER", "DAGSHUB_TOKEN", "GITHUB_PAT"]:
    os.environ[k] = s.get_secret(k)
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/SanniyaAslam/AwaazGuard.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USER"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
```
```python
# Cell 2
!git clone https://github.com/SanniyaAslam/AwaazGuard.git
%cd AwaazGuard
!pip install -q -r requirements.txt
!git config user.name "Your Name" && git config user.email "your-github-email"
!dvc remote modify origin --local access_key_id $DAGSHUB_TOKEN
!dvc remote modify origin --local secret_access_key $DAGSHUB_TOKEN
!dvc pull data/feedback
```
```python
# Cell 3: the whole pipeline. Expect roughly 1.5 to 3 hours. Keep the tab open.
!dvc repro
```
```python
# Cell 4: results, then upload data + models and push the results
!dvc metrics show
!cat reports/registry.json
!dvc push
!git add dvc.lock reports
!git commit -m "Pipeline run: ONNX model, metrics, registry record"
!git push https://$GITHUB_PAT@github.com/SanniyaAslam/AwaazGuard.git main
```

To save time, check `main.epochs` in `params.yaml` first (you used 3 in Deliverable 1; that is enough). If you
get a "disk full" error, attach an ASVspoof 2019 dataset to the notebook and set `ingest.local_dir`.

4. Make it the champion (laptop, `.env` filled in, or a Kaggle cell):
```powershell
python -m scripts.promote --list
python -m scripts.promote --version <N> --stage Production
python -m scripts.promote --version 1 --stage Archived
```
`<N>` is the new version from `--list`. Version 1 is the old Deliverable 1 model without ONNX files, so archive it.

**Check it worked:** `--list` shows your new version in **Production**, and DagsHub → Models shows the same.

Commit the pipeline results if Kaggle could not push (download `dvc.lock` and `reports/`, put them in your repo, then
`git add dvc.lock reports`, commit, push).

## Phase C: run the whole system on your laptop (S)

```powershell
docker run hello-world
copy deploy.env.example deploy.env
```
Open `deploy.env` in VS Code and fill in: `IMAGE=awaazguard`, `ADMIN_TOKEN` and `GRAFANA_ADMIN_PASSWORD` (long
random strings; make one with `python -c "import secrets; print(secrets.token_urlsafe(24))"`), and the three
`MLFLOW_TRACKING_*` values (same as your `.env`). Keep `PUBLIC_GRAFANA_URL` empty.

```powershell
docker compose --env-file deploy.env up -d --build
docker compose ps
```
The first build takes 5 to 15 minutes.

**Check it worked:** open `http://localhost/api/health`. `champion.loaded` must be `true`. Then open
`http://localhost`, upload a `.flac` from `data\processed\audio\eval`, and look at the result: verdict,
confidence percentage, per-second band, waveform and spectrogram. Answer the feedback question. Open
`http://localhost/dashboard/` (enter `ADMIN_TOKEN`): the check appears there.

If the app keeps restarting: `docker compose logs app`. The usual cause is wrong MLflow credentials or no model in Production.

## Phase D: Azure server and first deployment (F)

**Only start after commit 14 is pushed.** The server clones your GitHub repo when it boots, and the repo must be public.

1. Activate **Azure for Students** with your university email (azure.microsoft.com/free/students).
2. Install tools if needed: `winget install -e --id Microsoft.AzureCLI`, `winget install Hashicorp.Terraform`.
3. Log in and create an SSH key:
```powershell
az login
az account show --query id -o tsv          # copy the subscription id
mkdir $HOME\.ssh
ssh-keygen -t ed25519 -f $HOME\.ssh\awaazguard -C "awaazguard"      # press Enter twice
```
4. In your repo: `cd infra\terraform`, `copy terraform.tfvars.example terraform.tfvars`, and edit it:
```
subscription_id = "<from az account show>"
admin_cidr      = "<your IP from https://checkip.amazonaws.com>/32"
dns_label       = "awaazguard-fatima"
repo_url        = "https://github.com/SanniyaAslam/AwaazGuard.git"
budget_email    = "you@example.com"
```
5. Create everything:
```powershell
terraform init
terraform plan
terraform apply          # type yes; 2 to 4 minutes
terraform output
```
If it says the region or size is not allowed, set `location = "uaenorth"` (or `southeastasia`, `eastus`), or
`vm_size = "Standard_B1s"`, and apply again.

6. Wait 5 minutes, then check the server and put the secrets on it:
```powershell
ssh -i $HOME\.ssh\awaazguard azureuser@<public_ip>
cat /opt/awaazguard-ready          # prints: AwaazGuard host ready
cd /opt/awaazguard
cp deploy.env.example deploy.env
nano deploy.env                    # Ctrl+O, Enter, Ctrl+X to save and exit
```
In `deploy.env` on the server: `IMAGE=ghcr.io/sanniyaaslam/awaazguard` (all lowercase), long random
`ADMIN_TOKEN` and `GRAFANA_ADMIN_PASSWORD`, and the DagsHub `MLFLOW_TRACKING_*` values.

7. **Send Sanniya two things privately** (GitHub secrets can only be added by the repo owner): the output of
`terraform output public_ip`, and the **entire contents of the private key file** `$HOME\.ssh\awaazguard`
(the one without `.pub`). Sanniya deletes the message after adding it. It is a project-only key.

Your live URL is `http://<dns_label>.<region>.cloudapp.azure.com`. Keep a note of it.

Do **not** destroy the server until the TA has graded. `terraform destroy` removes everything when you are done.

## Phase E: CI/CD secrets, branch protection and proof that the gate blocks (S)

1. GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `VM_HOST` | the server's public IP |
| `VM_USER` | `azureuser` |
| `VM_SSH_KEY` | the whole private key text, including the BEGIN and END lines |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/SanniyaAslam/AwaazGuard.mlflow` |
| `DAGSHUB_USERNAME` | your DagsHub username |
| `DAGSHUB_TOKEN` | your DagsHub token |

Also **Settings → Actions → General → Workflow permissions → Read and write permissions**.

2. **First deploy:** Actions → **CD** → Run workflow. The first run pushes the image but the deploy step fails
(new GHCR images are private). Fix it once: GitHub profile → **Packages** → `awaazguard` → Package settings →
Change visibility → **Public**. Then **Re-run all jobs**.

**Check it worked:** open the live URL and `/api/health`. Upload a file; the result appears.

3. **Make the gate block merges.** Settings → **Branches** → Add branch protection rule for `main`: tick
"Require a pull request before merging" and "Require status checks to pass before merging", and select
`Lint + unit tests`, `Data validation (data/processed)` and `Model evaluation gate (challenger vs production)`.
(The checks appear in the list after CI has run once.)

4. **Prove the gate works.** You need a screenshot of a blocked pull request:
   1. Register a deliberately bad model as the challenger. On your laptop (this takes a few minutes and uses a tiny fake dataset):
```powershell
python scripts/make_mock_dataset.py --out mock_data/LA --per-split 150
dvc exp run -S ingest.local_dir=mock_data/LA -S main.epochs=1 -S main.max_batches_per_epoch=2 -S main.batch_size=4 -S baseline.n_components=8
```
   2. Check `python -m scripts.promote --list`: the new version is in **Staging** with a poor score.
   3. Make any small change on a branch and open a pull request:
```powershell
git switch -c test-gate
git commit --allow-empty -m "Trigger CI gate"
git push -u origin test-gate
```
   4. On GitHub, open the pull request. The **Model evaluation gate** job turns red and its summary says
   **BLOCKED: challenger EER ... is worse than champion**. **Take a screenshot** of the failed check and of the job summary.
   5. Clean up: archive the bad version and remove the mock data.
```powershell
python -m scripts.promote --version <bad N> --stage Archived
dvc exp remove --all
Remove-Item -Recurse -Force mock_data, data\raw\LA, data\processed, data\features, models
git checkout -- dvc.lock params.yaml
```
   Re-run the pull request's checks: they go green. Close the pull request.

Warning for step 4.5: `data\raw\LA` must be deleted, otherwise the next real run would silently use the mock data.
After cleaning up, run `dvc pull data/processed` again before using the data.

**Note on the model quality ceiling:** `params.yaml → gate.max_eer` is 0.35. If your champion's EER on the fixed set is above that,
the gate fails for every pull request. Look at the printed EER in the job summary and set `max_eer` a bit above it.

## Phase F: observability (both)

1. Grafana: `http://<public_ip>:3001`, user `admin`, password = `GRAFANA_ADMIN_PASSWORD`. Open
**Dashboards → AwaazGuard → AwaazGuard - Production**. (Port 3001 only accepts the IP in `admin_cidr`;
if your IP changed, update it and run `terraform apply`.)
2. Send traffic so the panels fill in:
```powershell
python -m scripts.simulate_traffic --gateway http://<public_ip> --source asvspoof --n 120 --feedback-rate 0.5
```
3. Look at: requests and latency p95 per variant, errors, verdicts, canary share, drift tiles. The Operations page
(`http://<public_ip>/dashboard/`) has the same plus the **Latest drift report** (Evidently).

The drift monitor runs every 15 minutes and needs at least 50 checks. While testing, set
`MONITOR_INTERVAL_MIN=2` in `deploy.env` on the server, then run
`docker compose --env-file deploy.env up -d` there.

## Phase G: closed loop: drift, retrain, challenger vs champion, promote or roll back

Do the slow parts (retraining) a day before the video; the video then shows the trigger and the results.

1. **Simulate drift.** ASVspoof eval clips often trigger drift on their own (they contain attacks the model never saw). For
a sure alarm, send phone-quality fakes from the red-team generator:
```powershell
pip install -r requirements-flows.txt
python -m piper.download_voices en_US-lessac-medium en_US-libritts_r-medium --data-dir voices
dvc pull data/processed
python -m src.redteam.run_redteam generate --generators piper --n-fake 200 --n-real 200
python -m scripts.simulate_traffic --gateway http://<public_ip> --source folder --dir data\redteam\latest --n 200
```
(The red-team batch folder has a `manifest.csv` with the file names and labels, which the script reads.)
After the next monitor cycle the drift tile turns red and the Operations page
says **drift detected**. If Piper will not install on Windows, run the generate command in a Kaggle notebook and download the `data/redteam/latest` folder.

2. **Trigger retraining** (Kaggle GPU notebook, with Cells 1 and 2 from Phase B and `pip install -r requirements-flows.txt`):
```python
!git remote set-url origin https://$GITHUB_PAT@github.com/SanniyaAslam/AwaazGuard.git
!python -m flows.retrain_flow --gateway http://<public_ip> --token <ADMIN_TOKEN> --generators piper --commit
```
Add `--force` to retrain without drift. The flow checks drift, exports feedback, builds a red-team batch, versions the data
with DVC, runs `dvc repro` (about 1.5 to 3 hours), runs the red-team gate, and starts a 10% canary. If Piper fails to install,
use `--generators stub`.

**Check it worked:** the flow ends with `canary_started`. The Operations page shows the new version as the candidate with 10% of
traffic, and DagsHub → Models shows it in **Staging** with a `redteam_gate=pass` tag.

3. **Challenger vs champion.** Send traffic with labels so both models get feedback:
```powershell
python -m scripts.simulate_traffic --gateway http://<public_ip> --source asvspoof --n 150 --feedback-rate 0.6
```
4. **Decision: promote or roll back** (any machine with `.env`):
```powershell
python -m flows.canary_flow --gateway http://<public_ip> --token <ADMIN_TOKEN>
```
It waits until the candidate has 50 requests, then **rolls back** if the candidate's p95 latency is over 1.5x the champion's or its feedback
EER is more than 0.02 worse, and otherwise **promotes** (old champion archived, no downtime). The decision and reason appear under
**Rollout history** on the Operations page.

If the retrained model happens to be worse, that is a rollback, which is also a valid result for the TA. Show whichever happens.

## Phase H: the 5-minute demo video (both)

Record with OBS Studio or Windows Game Bar (Win + G). Record each part separately and join them in any simple editor (Clipchamp is
free). Retraining is slow, so say clearly: "retraining was run earlier; here is the result". Do not pretend it ran live.

| Time | Who | Show | Say |
|---|---|---|---|
| 0:00 | F | Live URL; upload a real and a fake clip | The service is containerized on Azure. Verdict, confidence, per-second risk band (saliency), spectrogram. |
| 0:50 | S | Send feedback; Operations page; Grafana | Latency, error rate and verdicts are monitored live with Prometheus and Grafana. |
| 1:40 | S | Drift tile red; Evidently report | Production audio no longer looks like the training data. The drift alert fires from MMD, the out-of-distribution rate and Evidently. |
| 2:30 | F | Start `retrain_flow` (show the command and the first log lines), then the finished run from the day before: new Staging version in MLflow, red-team gate pass | The alert triggers retraining on feedback plus fresh fakes. The pipeline is reproducible with DVC. |
| 3:30 | S | GitHub pull request with the red **model gate** (screenshot or live) and the green CI run | CI blocks a model that is worse than production on a fixed test set. CD only deploys after CI passes. |
| 4:10 | F | Run `canary_flow`; Rollout history on the Operations page | The challenger got 10% of traffic and is compared with the champion. Promote or rollback, with the reason. |
| 4:50 | both | Summary | What we would add next. |

Rehearse once with a timer. Upload the video to YouTube (unlisted) or Google Drive (anyone with the link can view) and test the link in a private window.

## Phase I: submit (S)

- [ ] **GitHub repo link:** `https://github.com/SanniyaAslam/AwaazGuard` (public, both names in Contributors, CI green on `main`)
- [ ] **Live deployment URL:** `http://<dns_label>.<region>.cloudapp.azure.com` (open it in a private window; the server must be running)
- [ ] **5-minute demo video link** (with a live prediction, drift alert, retraining trigger and model promotion)
- [ ] Optional extras in the message: screenshot of the blocked pull request, the Grafana dashboard, and `README.md` section "CI gates".

Final repo check: no `.env`, `deploy.env`, `terraform.tfvars`, private key, `data/`, `models/` or tokens on GitHub.

---

## Changes from the handbook (this guide wins where they differ)

- The repo and DagsHub belong to **Sanniya**, so replace every `<fatima-github>` and `<github-username>` with `SanniyaAslam`, and use `AwaazGuard`
  with the capital letters. (The image name is lowercase: `ghcr.io/sanniyaaslam/awaazguard`.)
- The handbook's 18-commit plan assumes an empty repo. Your repo already has the Deliverable 1 commits, so Phase A above only adds the new files.
- The handbook CI ran the model check only on `main` with a fixed EER limit. Now every pull request runs unit tests, data validation and a
  **champion vs challenger gate** on a fixed test set, and CD runs only after CI succeeds.
- The demo is 5 minutes, not 10.
- Commands are PowerShell. `rm -rf` becomes `Remove-Item -Recurse -Force`, `cp` becomes `copy`.
- Only the repo owner can add GitHub secrets, so Fatima sends the server IP and the private key to Sanniya.
- Run the retraining flows on Kaggle (Linux): `prefect`, `httpx` and `piper-tts` are in `requirements-flows.txt`, which is separate from `requirements.txt`.

## If something goes wrong

| Problem | Fix |
|---|---|
| Azure for Students does not verify your email | Ask the university IT desk, or use the temporary-URL fallback: run Docker on a laptop and expose it with `cloudflared tunnel --url http://localhost`. Tell the TA the URL is temporary. |
| `RequestDisallowedByAzure` or SKU not available | Change `location` or `vm_size` in `terraform.tfvars` (see Phase D) |
| CI `data-validation` says secrets are missing | Add the six secrets in Phase E step 1 |
| `model-gate` says "No model in stage Production" | Promote a version (Phase B step 4) |
| `model-gate` fails for every pull request | Look at the EER in the job summary; raise `gate.max_eer` in `params.yaml` if the ceiling is too strict |
| Deploy fails with "denied" or "unauthorized" | Make the GHCR package public (Phase E step 2) |
| Microphone button does nothing | Browsers allow the microphone only on `https://` or `localhost`. Upload files instead, or add HTTPS (handbook appendix). |
| `git push` is rejected | `git pull`, then `git push` again |
