# SPEC 01 — Project foundation

> **Status:** Approved
> **Depends on:** —
> **Backlog:** E0 (Setup and serving spike), tasks E0.1–E0.5 of `BACKLOG.md`
> **Date:** 2026-10-30
> **Objective:** Leave a verified cluster, a git-tracked project skeleton, a pinned `churn-base` image and the raw CSVs on the `churn-data` PVC, so pipelines and the serving spike can start.

## Why this spec exists

E0 touched five areas (cluster, image, data, EDA, serving), so it was split in two.
This spec covers the foundation (E0.1–E0.5).
SPEC 02 covers the local baseline, the EDA notebook and the serving spike (E0.6–E0.7).
The E0 "done when" (a raw row returns probabilities from KServe) is only met after SPEC 02.

## Scope

**In:**

- **E0.1 Cluster check:** `minikube start`, every pod in `kubeflow` and `kserve` is `Running`, and the KFP UI loads through `kubectl port-forward`.
- **E0.2 KServe versions:** read the exact scikit-learn and Python versions inside the KServe 0.21 sklearnserver image and record them in `docs/runbook.md` and `DECISIONS.md`.
- **E0.3 Project skeleton:**
  - Create the folders of README §8: `docker/churn-base/`, `k8s/`, `components/`, `pipelines/`, `tools/`, `docs/`, `specs/`.
  - Run `git init` and make one local commit, with no remote.
  - Check that `.gitignore` excludes the CSVs.
- **Project memory files:**
  - `CLAUDE.md`: a short entry point that links to README, BACKLOG and DECISIONS.
  - `DECISIONS.md`: the frozen contracts (`FEATURES` order, dropped columns, target `Exited`, stratified 80/20 split with `SEED`, ROC-AUC as main metric, gate rule, model `churn` with alias `@champion`, sklearn pinned to the sklearnserver version).
- **Python environment with uv:**
  - `pyproject.toml` and `uv.lock` at the repo root are the single source of versions.
  - Runtime dependencies: scikit-learn (E0.2 version), mlflow, optuna, pandas, joblib.
  - A `dev` group holds jupyter and matplotlib and stays out of the image.
- **E0.4 `churn-base` image:**
  - `docker/churn-base/Dockerfile` uses a `python:3.X-slim` base whose Python version matches the sklearnserver (E0.2).
  - `docker/churn-base/requirements.txt` is generated with `uv export` and never edited by hand.
  - Build `churn-base:0.1.0` with `minikube image build`.
  - A test pod in `kubeflow` imports every runtime dependency.
- **E0.5 Data PVC:**
  - `k8s/pvc-churn-data.yaml`: 2Gi, `ReadWriteOnce`, storageClass `standard`, namespace `kubeflow`.
  - `k8s/data-loader-pod.yaml`: helper pod used for `kubectl cp` of the 3 CSVs to `/mnt/data/raw/`, deleted afterwards.
- **README fixes:** the data now lives in `data/`, so update §2 and §8.
- **`docs/runbook.md`:** started here with the commands actually used in this spec.

**Out of scope (for future specs):**

- Local EDA notebook and the `tools/` baseline (E0.6). Goes in SPEC 02.
- Serving spike with `PREDICT_PROBA` (E0.7). Goes in SPEC 02.
- Fallback for the spike (custom predictor or `FunctionTransformer`). Gets its own spec if the spike fails.
- Remote GitHub repo and push.
- MLflow, pipelines, KServe InferenceServices, monitoring (E1–E4).
- `components/config.py` and any Python code under `components/` or `pipelines/`.

## Data model

This spec introduces no code structures. It defines the PVC layout, the `pyproject.toml` shape, the image name and the frozen contracts in `DECISIONS.md`.

### PVC `churn-data` layout (namespace `kubeflow`, mounted at `/mnt/data`)

```text
/mnt/data/
└── raw/
    ├── train.csv              # 15,000 rows, includes Exited
    ├── test.csv               # 10,000 rows, ids 15000+
    └── sample_submission.csv  # id,Exited
```

Later specs add `processed/`, `models/churn/v<N>/` and `submission.csv`. This spec only creates `raw/`.

### `pyproject.toml` (shape, versions resolved by uv)

```toml
[project]
name = "churn-mlops"
version = "0.1.0"
requires-python = "==3.X.*"          # = sklearnserver Python (E0.2)
dependencies = [
  "scikit-learn==<E0.2 version>",    # exact pin, never a range
  "mlflow",
  "optuna",
  "pandas",
  "joblib",
]

[dependency-groups]
dev = ["jupyter", "matplotlib"]
```

- `uv.lock` pins every transitive version.
- `docker/churn-base/requirements.txt` = `uv export --no-dev --no-hashes --format requirements-txt`.

### Image

- Name and tag: `churn-base:0.1.0`.
- The tag goes up on every change to `pyproject.toml` or the `Dockerfile`.
- Pods use `imagePullPolicy: IfNotPresent`.

### Frozen contracts in `DECISIONS.md`

| Contract | Value |
|---|---|
| Target | `Exited` (0/1), output is a probability |
| Dropped columns | `id`, `CustomerId`, `Surname` (`id` kept aside only for the submission) |
| `FEATURES` order | `CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary` |
| Split | Stratified 80/20 train/hold-out, `SEED = 42` |
| Main metric | ROC-AUC (plus PR-AUC, recall, F1, confusion matrix) |
| Gate | Hold-out ROC-AUC > Dummy baseline + 0.05 **and** ≥ current champion's hold-out AUC |
| Registry | Model `churn`, alias `@champion` |
| scikit-learn | Pinned to the KServe 0.21 sklearnserver version (E0.2) |

Changing any row needs a new line in `DECISIONS.md` first.

## Implementation plan

1. **Skeleton and git.**
   - Create `docker/churn-base/`, `k8s/`, `components/`, `pipelines/`, `tools/`, `docs/` and `specs/`, each with a `.gitkeep` while it is empty.
   - Write `CLAUDE.md` (links to README, BACKLOG, DECISIONS).
   - Fix README §2 and §8 so they point to `data/`.
   - `git init`.
   - Check: `git status --ignored` lists `data/*.csv` as ignored and not as untracked.
   - First commit.
2. **Cluster check (E0.1).**
   - `minikube start`.
   - Check that `kubectl get pods -n kubeflow` and `kubectl get pods -n kserve` show every pod `Running` (or `Completed` for jobs).
   - `kubectl port-forward` to `ml-pipeline-ui` and open the UI.
   - Write the commands in `docs/runbook.md` under "Cluster".
3. **KServe versions (E0.2).**
   - Read the sklearnserver image from the `kserve-sklearnserver` ClusterServingRuntime.
   - Run that image with `python --version` and `pip show scikit-learn`.
   - Write the image, Python and scikit-learn versions to `docs/runbook.md`.
4. **`DECISIONS.md`.** Write the frozen contracts from the data model, with the sklearn version from step 3.
5. **Python environment.**
   - `pyproject.toml` with `requires-python` and the scikit-learn pin from step 3, the runtime dependencies and the `dev` group.
   - `uv lock` and `uv sync`.
   - Check: `uv run python -c "import sklearn, mlflow, optuna, pandas, joblib"` exits 0, and `sklearn.__version__` matches step 3.
   - Commit `pyproject.toml`, `uv.lock` and `.python-version`.
6. **`churn-base` image (E0.4).**
   - Generate `docker/churn-base/requirements.txt` with `uv export`.
   - `Dockerfile`: `python:3.X-slim` base, `pip install -r requirements.txt`, no `CMD`.
   - `minikube image build -t churn-base:0.1.0 docker/churn-base`.
   - Check: `kubectl run` in `kubeflow` with `--image-pull-policy=IfNotPresent` and `--rm` imports every runtime dependency and prints the sklearn version.
   - Write the build commands to the runbook.
7. **Data PVC (E0.5).**
   - Write `k8s/pvc-churn-data.yaml` and `k8s/data-loader-pod.yaml` (`busybox`, mounts `churn-data` at `/mnt/data`, `sleep` command).
   - Apply both.
   - `kubectl exec ... mkdir -p /mnt/data/raw`, then `kubectl cp` the 3 CSVs.
   - Check: `ls -l /mnt/data/raw` shows the 3 files, and `wc -l` gives 15001 / 10001 / 10001.
   - Delete the helper pod. The PVC stays.
   - Write the commands to the runbook.

## Acceptance criteria

### Cluster (E0.1)
- [ ] `kubectl get pods -n kubeflow` shows every pod `Running` or `Completed`, with no restarts loop (`CrashLoopBackOff`, `Error`).
- [ ] `kubectl get pods -n kserve` shows the KServe controller `Running`.
- [ ] The KFP UI loads in the browser through `kubectl port-forward`.

### KServe versions (E0.2)
- [ ] `docs/runbook.md` records the sklearnserver image reference, its Python version and its exact scikit-learn version.

### Skeleton and memory files (E0.3)
- [ ] The folders `docker/churn-base/`, `k8s/`, `components/`, `pipelines/`, `tools/`, `docs/` and `specs/` exist.
- [ ] `git log` shows at least one commit, and `git remote -v` is empty.
- [ ] `git ls-files | grep -c '\.csv$'` returns 0.
- [ ] `CLAUDE.md` exists and links to `README.md`, `BACKLOG.md` and `DECISIONS.md`.
- [ ] `DECISIONS.md` lists every contract of the data model table, including `SEED = 42` and the scikit-learn version from E0.2.
- [ ] README §2 and §8 point to `data/` for the CSVs.

### Python environment
- [ ] `pyproject.toml` pins scikit-learn with `==` to the E0.2 version, and `requires-python` matches the sklearnserver Python minor version.
- [ ] `uv.lock` is committed.
- [ ] `uv run python -c "import sklearn, mlflow, optuna, pandas, joblib; print(sklearn.__version__)"` exits 0 and prints the E0.2 version.
- [ ] `jupyter` and `matplotlib` appear only in the `dev` group.

### `churn-base` image (E0.4)
- [ ] `docker/churn-base/requirements.txt` matches the output of `uv export --no-dev --no-hashes --format requirements-txt`, and contains neither `jupyter` nor `matplotlib`.
- [ ] `minikube image ls` lists `churn-base:0.1.0`.
- [ ] A pod in `kubeflow` running `churn-base:0.1.0` imports `sklearn, mlflow, optuna, pandas, joblib` and prints the same scikit-learn version as E0.2.

### Data PVC (E0.5)
- [ ] `kubectl get pvc churn-data -n kubeflow` shows `Bound`, with 2Gi capacity.
- [ ] `/mnt/data/raw/` on the PVC contains `train.csv` (15001 lines), `test.csv` (10001 lines) and `sample_submission.csv` (10001 lines).
- [ ] `kubectl get pod data-loader -n kubeflow` returns `NotFound` after the copy.

### Runbook
- [ ] `docs/runbook.md` has a section per step (cluster, versions, environment, image, PVC) with the commands actually run.

## Decisions

- **Yes:** split E0 into SPEC 01 (foundation, E0.1–E0.5) and SPEC 02 (baseline and serving spike, E0.6–E0.7). E0 touched five areas, and the spike deserves its own criteria and decisions.
- **No:** one spec for all of E0. Too many areas for one contract.
- **Yes:** `pyproject.toml` + `uv.lock` as the single source of versions, with the image `requirements.txt` generated by `uv export`. The local `.venv` and the image then share the same versions.
- **No:** a hand-written `requirements.txt` kept apart from the local environment. The two would drift.
- **Yes:** jupyter and matplotlib in a `dev` group, kept out of the image. They are only for the local EDA notebook (SPEC 02).
- **Yes:** the Python version of `churn-base` matches the sklearnserver image (read in E0.2). A model pickled with one Python/sklearn pair must load under the same pair at serving time.
- **No:** fixing Python 3.11 up front. It could differ from the sklearnserver.
- **Yes:** versioned tag `churn-base:0.1.0` with `IfNotPresent`. With `latest`, minikube can serve a stale image without warning.
- **Yes:** CSVs stay in `data/`, and the README is fixed to match. The folder is tidier and `.gitignore` already covers `*.csv`.
- **No:** moving the CSVs to the repo root.
- **Yes:** PVC of 2Gi, `ReadWriteOnce`, storageClass `standard`. Enough for the data and a few model versions, and RWO works for several pods on a single-node minikube.
- **Yes:** `busybox` helper pod, deleted after the copy. It is light, does not depend on `churn-base`, and leaves no idle pod.
- **Yes:** `kubectl run --rm` for the image import test, with no manifest. The test runs once and needs no file.
- **Yes:** `SEED = 42`. The README asked for a fixed seed without a value, and 42 is the convention.
- **Yes:** `CLAUDE.md` and `DECISIONS.md` created in this spec. The spec workflow expects them, and the frozen contracts need a single place.
- **Yes:** `git init` with a local commit only. Publishing to GitHub is a separate step for later.
- **Yes:** skeleton and git first, before the cluster check. Every later step can then end in its own commit.
- **No:** a fallback for the serving spike in this spec. If the spike fails, the fallback gets its own spec.

## Risks

| Risk | Mitigation |
|---|---|
| The KServe sklearnserver image may be amd64-only, and the Mac is Apple Silicon. Reading its versions or serving it in SPEC 02 could fail or run under emulation. | E0.2 reads the image's platforms with `docker manifest inspect` and records them in the runbook. Run with `--platform linux/amd64` just to read the versions. If it is amd64-only, SPEC 02 must cover that before the spike. |
| The pinned scikit-learn version may not resolve together with recent mlflow, optuna or pandas. | Let `uv lock` pick older compatible versions. If that fails, pin the conflicting package too and record why in `DECISIONS.md`. |
| `minikube start` with 6 GB may leave KFP pods in `Pending`, `OOMKilled` or `CrashLoopBackOff`. | Wait for the startup to settle and check `kubectl describe` / events. If memory is the cause, apply the README fallback (Docker Desktop ~10 GB, `--memory=8192`). |
| The PVC uses minikube's hostpath, so `minikube delete` loses the data. | The CSVs stay in `data/`, and the runbook reloads them in a few commands. |
| `minikube image build` builds inside minikube, not in the host Docker. The image is not visible to `docker images` on the Mac. | Check with `minikube image ls`, as the acceptance criteria already do. |

## What is **not** in this spec

- Local EDA notebook and `tools/` baseline (SPEC 02).
- Serving spike with `PREDICT_PROBA` (SPEC 02).
- Any serving fallback (own spec, only if the spike fails).
- Remote GitHub repo and push.
- MLflow, pipelines, InferenceServices, monitoring (E1–E4).
- `components/config.py` or any pipeline code.

Each one of those, if it lands, goes in its own spec.
