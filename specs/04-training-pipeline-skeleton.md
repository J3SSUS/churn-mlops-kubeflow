# SPEC 04 — Training pipeline skeleton (config, ingest, preprocess)

> **Status:** Approved
> **Depends on:** SPEC 01, SPEC 02, SPEC 03
> **Backlog:** E2 (Pipeline 1 — Training), tasks E2.1–E2.3 of `BACKLOG.md`, plus the first part of E2.8 (wiring with two steps) and the `churn-base:0.2.0` image
> **Date:** 2026-10-31
> **Objective:** Run on the cluster a KFP pipeline with the Ingest and Preprocess steps that validates `train.csv` and leaves a stratified train/hold-out split on the PVC, using shared config, `churn-base:0.2.0` and code mounted from a ConfigMap.

## Why this spec exists

E2 is too big for one spec: nine tasks that touch config, five components, the pipeline, MLflow and the gate. It is split in three, and this is the first one.

SPEC 04 builds the base that the other two stand on: the config, the first two steps and a KFP run that works on this cluster. It is the first pipeline that runs here, so the risk is in the infrastructure, not in the model.

Three things are not obvious:

- `churn-base` gets a new tag, `0.2.0`. KFP lightweight components run `kfp` inside the pod, and `0.1.0` does not carry it.
- The shared code (`config.py`, `preprocessing.py`) does not go in the image. It travels in a ConfigMap, as in SPEC 02, so a code change needs no rebuild.
- No MLflow call happens here. The MLflow constants enter `config.py` because SPEC 03 promised it for E2.1, and SPEC 05 is the first to use them.

The split of E2 is:

| Spec | Tasks | Leaves |
|---|---|---|
| SPEC 04 (this) | E2.1, E2.2, E2.3, first part of E2.8 | A green run of Ingest → Preprocess and the split on the PVC |
| SPEC 05 | E2.4, E2.5 | Tune with Optuna, nested MLflow runs and Evaluate |
| SPEC 06 | E2.6, E2.7, E2.9 | Gate, Register, `churn@champion` and the second run |

## Scope

**In:**

- **E2.1 Shared config:** extend `components/config.py` with the MLflow tracking URI, the experiment name, the model name `churn`, the alias `champion`, the processed-data directory on the PVC and the hold-out fraction. `FEATURES`, the indices, `SEED` and the PVC paths already exist from SPEC 02.
- **E2.2 Ingest component:** reads `train.csv` from the PVC and fails the step if the data breaks the contract.
  - Columns `id`, the 10 `FEATURES` and `Exited` are present.
  - Numeric columns have numeric dtypes, and `Geography` and `Gender` are text.
  - No nulls in any column (0 %).
  - `Exited` only has the values 0 and 1.
- **E2.3 Preprocess component:** drops `id`, `CustomerId` and `Surname`, makes the stratified 80/20 split with `SEED`, and writes `train.csv` and `holdout.csv` to `/mnt/data/processed` on the PVC, with the columns in `FEATURES` order plus `Exited`.
- **`build_preprocessor()`** in `components/preprocessing.py`: one-hot by position for `CATEGORICAL_IDX` and scaling for `NUMERIC_IDX`. It returns an **unfitted** transformer. SPEC 05 (Tune) and SPEC 06 (Register) use it. It is checked locally here, and no model is trained or registered.
- **Image `churn-base:0.2.0`:** `kfp==2.15.0` (same as the KFP backend) added to the runtime dependencies and the image rebuilt. The user runs the build. `kfp-kubernetes` goes in the `dev` group because it is only needed to compile.
- **Shared code on the cluster:** ConfigMap `churn-code` in `kubeflow` with `config.py` and `preprocessing.py`, mounted at `/code` and added to `PYTHONPATH`, plus a script that refreshes it before each run.
- **E2.8 first part:** `pipelines/training_pipeline.py` with Ingest → Preprocess only, `churn-base:0.2.0` as base image, the `churn-data` PVC mounted with `kfp-kubernetes`, and the ConfigMap mounted at `/code`. A small script compiles it and submits it with `kfp.Client` through a port-forward.
- **Run and check:** the run is green in the KFP UI, and `train.csv` and `holdout.csv` exist on the PVC with the expected row counts (12,000 / 3,000).
- **Docs:** a section in `docs/runbook.md`, and lines in `DECISIONS.md` for the `kfp` pin and the new image tag.

**Out of scope (for future specs):**

- Tune with Optuna, nested MLflow runs and `ParallelFor` (SPEC 05, E2.4).
- Evaluate against the Dummy baseline (SPEC 05, E2.5).
- Gate, Register, the alias `champion` and the second run (SPEC 06, E2.6, E2.7, E2.9).
- Any MLflow call in the pipeline steps. The constants are defined here, but nothing uses them yet.
- Validating the categories of `Geography` and `Gender`, the row count and the share of positives in Ingest.
- Parquet, and a per-run folder for the split.
- Pipeline 2 (deploy), monitoring and the load generator.

## Data model

This spec adds no business data. It defines the shared constants, the PVC layout, the cluster objects and the contract of each component.

### Constants added to `components/config.py`

```python
MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:5000"
MLFLOW_EXPERIMENT = "churn-training"
MODEL_NAME = "churn"                 # frozen contract
CHAMPION_ALIAS = "champion"          # frozen contract
PROCESSED_DIR_PVC = "/mnt/data/processed"
HOLDOUT_FRACTION = 0.2
```

`FEATURES`, `CATEGORICAL_IDX`, `NUMERIC_IDX`, `TARGET`, `ID_COL`, `SEED`, `RAW_DIR_PVC` and `MODELS_DIR_PVC` already exist and do not change. Nothing in this spec reads the MLflow constants yet.

### PVC layout (`churn-data`, mounted at `/mnt/data`)

```text
/mnt/data/
├── raw/                 # existing: train.csv, test.csv, sample_submission.csv
└── processed/           # new, written by Preprocess, overwritten on every run
    ├── train.csv        # 12,000 rows: the 10 FEATURES in order + Exited
    └── holdout.csv      # 3,000 rows, same columns
```

- Both files have a header and no index column, and keep `Geography` and `Gender` as text.
- `id`, `CustomerId` and `Surname` are not in the processed files.
- The split is `train_test_split(test_size=HOLDOUT_FRACTION, stratify=y, random_state=SEED)`, so it is the same on every run.

### Components (contract)

| Component | Inputs | Output | Fails when |
|---|---|---|---|
| `ingest` | none (reads `RAW_DIR_PVC/train.csv`) | the raw file path (`str`) | a column is missing, a dtype is wrong, any null exists, or `Exited` has a value other than 0 or 1 |
| `preprocess` | the raw file path from `ingest` | `train_path`, `holdout_path`, `n_train`, `n_holdout` | the file cannot be read or written |

- Each component lives in its own file under `components/` and is a `@dsl.component(base_image="churn-base:0.2.0")` function.
- Imports of `config` and `preprocessing` go **inside** the function body, because KFP serializes only the function and the modules come from `/code`.
- Every task sets `PYTHONPATH=/code`.

### `build_preprocessor()` (`components/preprocessing.py`)

```python
ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_IDX),
    ("num", StandardScaler(), NUMERIC_IDX),
])
```

It returns the transformer **unfitted**. It is the same transformer as the throwaway one in `tools/spike_pipeline.py`, and columns are selected by position.

### Cluster objects

| Object | Namespace | Notes |
|---|---|---|
| Image `churn-base:0.2.0` | — | `0.1.0` plus `kfp==2.15.0`, built with `minikube image build` by the user |
| ConfigMap `churn-code` | `kubeflow` | Keys `config.py` and `preprocessing.py`, mounted at `/code`, refreshed by a script before each run |
| PVC `churn-data` | `kubeflow` | Existing, mounted at `/mnt/data` |
| Pipeline `churn-training` | `kubeflow` | Compiled from `pipelines/training_pipeline.py`, two steps for now |

## Implementation plan

1. **Decision lines and dependencies (image prerequisites).**
   - Add two lines to the `DECISIONS.md` log: `kfp` pinned to `2.15.0` (same as the KFP backend), and the image tag `churn-base:0.2.0` (`0.1.0` plus `kfp`).
   - In `pyproject.toml`, add `kfp==2.15.0` to the runtime dependencies and `kfp-kubernetes` to the `dev` group (pin the version that matches `kfp 2.15.0`). Run `uv lock` and regenerate `docker/churn-base/requirements.txt` with the `uv export` command in its header.
   - Check: `git diff docker/churn-base/requirements.txt` adds only `kfp` and its dependencies (and moves `protobuf` from 7.36.2 to 6.33.6), `scikit-learn` is still `1.5.2`, and `uv run python -c "import kfp; from kfp import kubernetes"` exits 0.
2. **Build the image `churn-base:0.2.0` (user runs it).**
   - The user runs `minikube image build` with the new tag, with the same command as in the runbook.
   - Check: `minikube image ls | grep churn-base` shows `0.1.0` and `0.2.0`, and a temporary pod with `0.2.0` prints the versions of `kfp` (2.15.0), `scikit-learn` (1.5.2) and `mlflow` (3.17.2). Delete the pod afterwards.
3. **Shared config (E2.1).**
   - Add the constants of the data model to `components/config.py`.
   - Check: `uv run python -c "import sys; sys.path.insert(0, 'components'); import config; print(config.MODEL_NAME, config.PROCESSED_DIR_PVC)"` prints `churn /mnt/data/processed`, and `FEATURES` is unchanged.
4. **`build_preprocessor()` (E2.3, helper).**
   - Write `components/preprocessing.py`.
   - Check, locally on `data/train.csv`: the returned transformer has no `transformers_` attribute (unfitted), and inside a `Pipeline` with `LogisticRegression` it fits and `predict_proba` returns shape `(n, 2)`.
5. **Ingest component (E2.2).**
   - Write `components/ingest.py`.
   - Check, locally with `RAW_DIR_PVC` pointed at `data/`: it passes on `train.csv`. On a temporary copy with one null, one wrong dtype, a missing column and an `Exited = 2`, each case raises. The temporary copies live outside the repo.
6. **Preprocess component (E2.3).**
   - Write `components/preprocess.py`.
   - Check, locally with the paths pointed at a temporary folder: `train.csv` has 12,000 rows and `holdout.csv` 3,000, the columns are the 10 `FEATURES` in order plus `Exited`, `id`, `CustomerId` and `Surname` are absent, and the share of positives differs by less than 0.1 points between the two files.
7. **Pipeline wiring and code sync (E2.8, first part).**
   - Write `pipelines/training_pipeline.py` (Ingest → Preprocess, base image `churn-base:0.2.0` with `IfNotPresent`, `churn-data` mounted at `/mnt/data`, ConfigMap `churn-code` mounted at `/code`, `PYTHONPATH=/code`, caching off, small resource requests), `tools/sync_code.sh` (refreshes the ConfigMap) and `tools/submit_pipeline.py` (runs the sync, compiles and submits with `kfp.Client`). The compiled YAML goes in `.gitignore`.
   - Check: the pipeline compiles, and the YAML references the PVC, the ConfigMap and the image `churn-base:0.2.0`.
8. **Submit and verify on the cluster.**
   - Open `kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888` and run `tools/submit_pipeline.py`.
   - Check: the run is green in the KFP UI, the logs of both steps show no error, and a temporary pod from `k8s/data-loader-pod.yaml` lists `processed/train.csv` (12,001 lines) and `processed/holdout.csv` (3,001 lines). Delete that pod afterwards.
   - If a step fails on the image, the PVC or the ConfigMap, save the pod events as evidence and stop.
9. **Docs.**
   - Write the "Training pipeline skeleton" section of `docs/runbook.md` with the commands actually run, the observed results and the gotchas. Update the `churn-base` row of README §4 to the installed `0.2.0`.
   - Check: every command in the runbook section appears in the steps above.

## Acceptance criteria

### Image and dependencies
- [ ] `pyproject.toml` has `kfp==2.15.0` in the runtime dependencies and `kfp-kubernetes` in the `dev` group, and `uv run python -c "import kfp; from kfp import kubernetes"` exits 0.
- [ ] `git diff main -- docker/churn-base/requirements.txt` adds `kfp` and its dependencies, and `scikit-learn` is still `1.5.2`.
- [ ] `minikube image ls | grep churn-base` lists `churn-base:0.1.0` and `churn-base:0.2.0`.
- [ ] A pod with `churn-base:0.2.0` prints `kfp 2.15.0`, `scikit-learn 1.5.2` and `mlflow 3.17.2`.

### Config and preprocessing helper
- [ ] `components/config.py` defines `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`, `MODEL_NAME = "churn"`, `CHAMPION_ALIAS = "champion"`, `PROCESSED_DIR_PVC` and `HOLDOUT_FRACTION = 0.2`, and `FEATURES` has the same order as in `DECISIONS.md`.
- [ ] `build_preprocessor()` returns an unfitted transformer (no `transformers_` attribute) that fits inside a `Pipeline` with `LogisticRegression` and gives `predict_proba` of shape `(n, 2)`.

### Components
- [ ] Ingest passes on `data/train.csv`, and raises on each of these: a missing column, a wrong dtype, one null, and `Exited = 2`.
- [ ] Preprocess writes `train.csv` with 12,000 rows and `holdout.csv` with 3,000 rows, with the 10 `FEATURES` in order plus `Exited`.
- [ ] The processed files have no `id`, `CustomerId` or `Surname`, and the share of positives differs by less than 0.1 points between them.
- [ ] Two runs of Preprocess give identical files (same `SEED`).

### Pipeline on the cluster
- [ ] `kubectl get cm churn-code -n kubeflow` exists and holds the keys `config.py` and `preprocessing.py`.
- [ ] The compiled pipeline YAML references the image `churn-base:0.2.0`, the PVC `churn-data` and the ConfigMap `churn-code`.
- [ ] A run submitted with `tools/submit_pipeline.py` finishes `Succeeded` in the KFP UI, with both steps green.
- [ ] A temporary pod from `k8s/data-loader-pod.yaml` shows `processed/train.csv` with 12,001 lines and `processed/holdout.csv` with 3,001 lines. The pod is deleted afterwards.
- [ ] No `mlflow` run or experiment was created by this spec: `connectivity-test` is the only experiment besides the default one, and the registry is empty.
- [ ] `kubectl get pods -A --no-headers | awk '$4!="Running" && $4!="Completed"'` prints nothing, and no temporary pod is left in `kubeflow`.

### Docs and hygiene
- [ ] The `DECISIONS.md` log has dated lines for the `kfp` pin and the `0.2.0` tag, and `git diff main -- DECISIONS.md` shows no change to the frozen contracts table.
- [ ] `docs/runbook.md` has a "Training pipeline skeleton" section with the commands run, the observed results and the gotchas.
- [ ] The `churn-base` row of README §4 no longer says "to build".
- [ ] The compiled pipeline YAML is git-ignored, and `git ls-files | grep -c '\.csv$'` returns 0.

## Decisions

- **Yes:** split E2 into three specs: this one (E2.1–E2.3 plus the first part of E2.8), SPEC 05 (E2.4–E2.5) and SPEC 06 (E2.6, E2.7, E2.9). E2 has 9 tasks and touches config, five components, the pipeline, MLflow and the gate. Each spec leaves something that runs.
- **No:** all of E2 in one spec. It would be about 12 steps with many points of failure, against the rule that a spec fits in one sentence.
- **No:** two specs. The first would hold everything up to Evaluate, about 2 h, and the first cluster run of Optuna would come too late to find infrastructure problems.
- **Yes:** this spec ends with a green KFP run of Ingest and Preprocess only. It checks the PVC mount, the ConfigMap, the image and the permissions before any Optuna code exists.
- **Yes:** rebuild the image as `churn-base:0.2.0` with `kfp==2.15.0`. KFP lightweight components run `kfp.dsl.executor_main` inside the pod, so `kfp` must be in the image. It keeps the README rule "no `pip install` at every step" and gives the change a new, traceable tag.
- **No:** `packages_to_install=['kfp==...']` in every component. Each step would download packages at start, slower and dependent on internet.
- **No:** container components. They avoid the `kfp` dependency but lose the typed parameters and artifacts that Tune and Evaluate need.
- **Yes:** `kfp` pinned to `2.15.0`, the version of the KFP backend. `uv pip compile` resolves it with `mlflow-skinny 3.17.2`, `scikit-learn 1.5.2` and `numpy 2.4.6`. The only change is `protobuf` 7.36.2 → 6.33.6.
- **Yes:** `kfp-kubernetes` in the `dev` group only. It is needed to compile the pipeline, not to run the steps.
- **Yes:** shared code (`config.py`, `preprocessing.py`) through the ConfigMap `churn-code` mounted at `/code`. It is the SPEC 02 pattern, and a code change needs no rebuild and no image tag.
- **No:** code baked into the image. Every change would need a manual build and a new tag.
- **No:** code on the `churn-data` PVC. It mixes code with data and is easy to forget to update.
- **Yes:** the split goes to `/mnt/data/processed` as CSV, with a fixed path overwritten on every run. The split is deterministic (`SEED = 42`), so the hold-out is the same across runs, as the README asks.
- **No:** Parquet. It would need `pyarrow` in the image and in `pyproject`.
- **No:** a folder per run. Each run would add copies to a 2Gi PVC, and the path would have to travel between steps.
- **Yes:** Ingest fails on a missing column, a wrong dtype, any null and a target outside {0, 1}. Nulls are strict (0 %) because `train.csv` has none today, so a tolerance would only hide a change in the data.
- **No:** validating the categories of `Geography` and `Gender`, the row count or the share of positives. The user left them out. They would make Ingest fail on a legitimate change of the dataset.
- **Yes:** `build_preprocessor()` lives in this spec, as the backlog asks for E2.3, and returns an **unfitted** transformer. Fitting happens inside each CV fold (SPEC 05) and at Register (SPEC 06).
- **Yes:** `handle_unknown="ignore"` in the one-hot encoder, copied from the throwaway preprocessor of SPEC 02. A new category at serving time does not break the prediction.
- **Yes:** the MLflow constants (URI, experiment `churn-training`, model `churn`, alias `champion`) enter `components/config.py` here, as SPEC 03 promised for E2.1. **No** step uses them yet.
- **Yes:** keep `churn-base:0.1.0` in minikube. The pods of SPEC 02 and SPEC 03 used it, and removing it is out of scope.
- **Yes:** `ingest` returns the raw file path and `preprocess` takes it as input. It makes KFP treat them as dependent steps, even though `preprocess` reads the CSV from the PVC again.
- **Yes:** the compiled pipeline YAML is git-ignored. It is a generated file, rebuilt by `tools/submit_pipeline.py`.
- **No:** a negative test on the cluster with a broken file on the PVC. Spoiling the PVC to prove the failure has more risk than value. The local check of step 5 covers each failure case.
- **No:** `kfp` or `kfp-kubernetes` in the frozen contracts table. They are tools, and a log line in `DECISIONS.md` is enough.

## Risks

| Risk | Mitigation |
|---|---|
| Adding `kfp` moves `protobuf` from 7.36.2 to 6.33.6 in the image. `mlflow-skinny` or another package might misbehave with the older version. | Step 2 prints the versions and imports `mlflow` in a pod with `0.2.0`. If something breaks, pin `protobuf` and log it in `DECISIONS.md` before rebuilding. |
| The `kfp-kubernetes` version may not match `kfp 2.15.0` (compile errors or missing functions such as `mount_pvc` and the ConfigMap volume). | Step 1 pins the version that `uv` resolves together with `kfp==2.15.0`, and `from kfp import kubernetes` is checked there (the package `kfp-kubernetes` is imported as `kfp.kubernetes`). |
| KFP serializes only the component function. A module-level `import config` works at compile time on the laptop and fails in the pod with `ModuleNotFoundError`. | Imports of `config` and `preprocessing` go inside the function body (see Data model). Step 8 shows the error in the step log if it slips through. |
| KFP caching reuses the output of a step when its spec has not changed, even if the code in the ConfigMap did. A fixed bug could seem not to run. | The cache key does not include the ConfigMap content. Every task sets `set_caching_options(False)`, and the runbook states it. |
| The ConfigMap `churn-code` is stale and the pod runs old code. | `tools/submit_pipeline.py` runs `tools/sync_code.sh` before submitting, so the ConfigMap is refreshed on every run. |
| The first KFP run on this cluster may fail for reasons outside the code: PVC mount, image not found in minikube, or the step being unable to write to `/mnt/data/processed`. | Step 8 saves the pod events as evidence and stops. The image has no `USER`, so the pods run as root, as in SPEC 02, which makes a permission error unlikely. |
| The node is tight (about 8 GB, with KFP, KServe and MLflow). In SPEC 03 it hit load above 60 and `kubectl` timed out. | The two steps are light (15,000 rows) and get small requests and limits. The acceptance criteria check that no pod leaves `Running`. If the API hangs, use the diagnosis in the MLflow runbook section. |
| Preprocess overwrites `processed/` on every run, so two simultaneous runs would race. | One run at a time. With a single user and a deterministic split, the content would be identical anyway. |
| The strict ingest check (0 % nulls) stops the pipeline if the dataset ever changes. | Accepted. It is the purpose of the check, and the failure message names the column. |

## What is **not** in this spec

- Tune with Optuna, nested MLflow runs and `ParallelFor` (SPEC 05).
- Evaluate against the Dummy baseline (SPEC 05).
- Gate, Register, the alias `champion` and the second run (SPEC 06).
- Any MLflow call in the pipeline steps.
- Validating the categories of `Geography` and `Gender`, the row count and the share of positives in Ingest.
- Parquet and a folder per run for the split.
- Removing `churn-base:0.1.0` from minikube.
- Pipeline 2 (deploy), RBAC, monitoring and the load generator.

Each one of those, if it lands, goes in its own spec.
