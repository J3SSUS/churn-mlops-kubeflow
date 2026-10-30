# SPEC 02 — Baseline, EDA and serving spike

> **Status:** Implemented.
> **Depends on:** SPEC 01
> **Backlog:** E0 (Setup and serving spike), tasks E0.6–E0.7 of `BACKLOG.md`; E2.1 pulled forward in a minimal form
> **Date:** 2026-10-30
> **Objective:** Measure a local baseline on the churn data and verify, on a temporary KServe InferenceService, how a sklearn `Pipeline` trained in `churn-base` answers a raw row.

## Why this spec exists

SPEC 01 left the cluster, the `churn-base:0.1.0` image and the CSVs on the `churn-data` PVC ready.
This spec covers the two remaining tasks of E0: a local EDA and baseline (E0.6) and the serving spike (E0.7).
The spike checks the riskiest serving assumptions before any pipeline is built.

Two things differ from the backlog:

- A minimal `components/config.py` is pulled forward from E2.1, because the baseline and the spike both need the frozen `FEATURES` order.
- The EDA also produces a written summary with the key charts, which the backlog does not list.

The E0 "done when" (a raw row returns probabilities from KServe) is only met if the spike answers `yes` to its first two questions. This spec is met with the evidence recorded, whatever the answers are.

## Scope

**In:**

- **E2.1 pulled forward (minimal):** `components/config.py` with `FEATURES` in the frozen order, categorical and numeric column indices, `TARGET`, `SEED = 42` and the PVC paths. Baseline and spike import it, so the contract lives in one place. E2.1 later adds the MLflow URI and the experiment and model names.
- **E0.6 EDA notebook:** `notebooks/eda.ipynb`, with charts and a short interpretation after each one.
  - Charts: target balance, numeric distributions, and churn rate by Geography, Gender, Age band, NumOfProducts, IsActiveMember and zero balance.
  - The notebook is committed with its outputs (charts and aggregated tables) and no dump of raw rows. Markdown is in English.
  - The notebook itself saves the 4–6 most important charts as PNG in `docs/img/eda/`.
- **EDA summary:** `docs/eda-summary.md`, in English, written after the notebook runs: a short dataset snapshot and 4–6 key findings, each with its numbers and an embedded saved chart. Linked from the README.
- **E0.6 baseline:** `tools/baseline.py` runs 5-fold stratified CV (`SEED`) on `data/train.csv` and prints the ROC-AUC of `DummyClassifier(strategy="prior")` and a plain `LogisticRegression` behind a position-based `ColumnTransformer`. The numbers go into a "Baseline results" section of the README.
- **E0.7 serving spike:**
  - A training script run in a pod with `churn-base:0.1.0`: reads `/mnt/data/raw/train.csv`, fits the small `Pipeline`, writes `/mnt/data/models/spike/v1/model.joblib`.
  - A temporary InferenceService `churn-spike` in `kubeflow`: Standard mode, runtime `kserve-sklearnserver` set explicitly, `PREDICT_PROBA=true`, `storageUri: pvc://churn-data/models/spike/v1`.
  - A query of raw `test.csv` rows through `port-forward`, in `FEATURES` order, compared with the local `predict_proba` of the same model.
  - Cleanup: delete the InferenceService, the temporary pods, the ConfigMap and `models/spike/`.
- **Spike findings in `docs/runbook.md`:** one yes/no answer with evidence for each of the four questions (probabilities, raw mixed types, load with the numpy/Python patch difference, Standard mode with the explicit runtime). The answer may be "no".
- **README:** add the "Baseline results" section, link `docs/eda-summary.md`, add `notebooks/` to the §8 layout, and update the two spike risks in §7 with the verified result.

**Out of scope (for future specs):**

- Fallback if the spike fails (custom predictor or `FunctionTransformer`). Gets its own spec.
- `/metrics` check on the predictor (E4.1).
- The final `build_preprocessor()` component (E2.3). The preprocessing here is throwaway, defined in `tools/`.
- Hyperparameter tuning, more model families, feature engineering and train vs test drift analysis.
- MLflow, KFP pipelines, the final `churn` InferenceService and the load generator (E1–E4).
- Running the whole `test.csv` through the endpoint and building `submission.csv` (E4.4).

## Data model

### `components/config.py` (shape)

```python
FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CATEGORICAL_IDX = [1, 2]                      # Geography, Gender
NUMERIC_IDX = [0, 3, 4, 5, 6, 7, 8, 9]        # includes the 0/1 columns HasCrCard, IsActiveMember
TARGET = "Exited"
ID_COL = "id"
SEED = 42
RAW_DIR_LOCAL = "data"
RAW_DIR_PVC = "/mnt/data/raw"
MODELS_DIR_PVC = "/mnt/data/models"
```

- `FEATURES` repeats the frozen order of `DECISIONS.md`; the file is the only place in code that spells it out.
- Columns are always addressed by position, never by name.

### Throwaway pipeline (in `tools/`, shared by baseline and spike)

```python
Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_IDX),
        ("num", StandardScaler(), NUMERIC_IDX),
    ])),
    ("clf", LogisticRegression(max_iter=1000)),
])
```

- Input is a 2-D array in `FEATURES` order with no column names, like what the server passes.
- The baseline fits it inside each CV fold. The spike fits it once on all 15,000 rows.
- It is not the final `build_preprocessor()` of E2.3.

### EDA outputs in the repo

```text
docs/
├── eda-summary.md
└── img/eda/                   # 4–6 PNG files saved by the notebook, kebab-case topic names
    ├── target-balance.png
    ├── churn-by-geography.png
    └── ...                    # final list chosen after the notebook runs
```

The notebook is `notebooks/eda.ipynb`. Only aggregated charts are saved, never raw rows.

### PVC `churn-data` layout during the spike

```text
/mnt/data/
├── raw/                       # from SPEC 01
└── models/spike/v1/
    └── model.joblib           # the only file in the folder, temporary
```

`sklearnserver` accepts exactly one `.joblib`, `.pkl` or `.pickle` file per model folder (read from the image's `model.py`), so nothing else goes in `v1/`.

### Spike request and response (KServe v1 protocol)

```json
{"instances": [[649.0, "Germany", "Male", 37.0, 3.0, 106534.4, 1.0, 1.0, 1.0, 131173.9]]}
```

- Expected response: `{"predictions": [[p0, p1]]}`, because with `PREDICT_PROBA=true` the server returns the full `predict_proba` matrix. The churn probability is `p1`.
- The spike records the real shape of the answer. E3.4 and E4.4 depend on it.

### InferenceService `churn-spike` (shape)

```yaml
metadata: {name: churn-spike, namespace: kubeflow}
spec:
  predictor:
    model:
      modelFormat: {name: sklearn}
      runtime: kserve-sklearnserver
      storageUri: pvc://churn-data/models/spike/v1
      env: [{name: PREDICT_PROBA, value: "true"}]
```

- The cluster's default mode is already `Standard` (`inferenceservice-config`), so no mode annotation is needed.
- Endpoint: `/v1/models/churn-spike:predict` through `kubectl port-forward`.

### Spike findings record (`docs/runbook.md`)

| Question | Answer (yes/no) | Evidence |
|---|---|---|
| `PREDICT_PROBA=true` returns probabilities | | response body |
| A raw row with strings and numbers keeps its types and is scored | | response body, or the error |
| The model loads and matches the local `predict_proba` despite the numpy/Python patch differences | | max absolute difference |
| It runs in Standard mode with the explicit `kserve-sklearnserver` runtime | | pod status, runtime shown by `kubectl get isvc` |

## Implementation plan

1. **Shared config and pipeline.**
   - Create `components/config.py` (data model) and `tools/spike_pipeline.py` with `build_pipeline()`.
   - Add `spike-reference.json` to `.gitignore` (it will hold `test.csv` rows).
   - Modules are imported with `PYTHONPATH=components:tools`, locally and in the pod.
   - Check: `PYTHONPATH=components:tools uv run python -c "from config import FEATURES; from spike_pipeline import build_pipeline; print(len(FEATURES)); print(build_pipeline())"` prints `10` and the pipeline.
2. **Baseline (E0.6).**
   - `tools/baseline.py` reads `data/train.csv`, builds `X` in `FEATURES` order with no column names, and runs 5-fold stratified CV with `SEED`.
   - It prints mean and std ROC-AUC for `DummyClassifier(strategy="prior")` and for `build_pipeline()`.
   - Check: `PYTHONPATH=components:tools uv run python tools/baseline.py` exits 0, Dummy is about 0.5 and LogReg is higher.
   - Add the printed numbers to a "Baseline results" section of the README.
3. **EDA notebook, part 1.**
   - `notebooks/eda.ipynb`: load data, target balance, numeric distributions, each chart followed by a markdown interpretation in English.
   - No cell prints raw rows (`head()`, `sample()`).
   - Check: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb` finishes without errors.
4. **EDA notebook, part 2.**
   - Add churn rate by Geography, Gender, Age band, NumOfProducts, IsActiveMember and zero balance, with an interpretation after each chart.
   - Save the key charts from the notebook with `fig.savefig(..., dpi=150, bbox_inches="tight")` into `docs/img/eda/`.
   - Re-run `nbconvert --execute --inplace` so the committed notebook has its outputs.
   - Add `notebooks/` to the README §8 layout.
   - Check: the notebook runs top to bottom with no errors, a search of its outputs finds no table of raw rows, and `grep -c '/Users/' notebooks/eda.ipynb` returns 0.
5. **EDA summary.**
   - Choose the 4–6 charts that carry the findings and keep only those PNG files.
   - Write `docs/eda-summary.md`: dataset snapshot (rows, positive rate), then one finding per chart with its numbers, what it implies for the model, and the embedded image.
   - Link it from the README.
   - Check: every image path in the summary exists, and no PNG in `docs/img/eda/` is missing from the summary.
6. **Spike training in a `churn-base` pod.**
   - `tools/spike_train.py` fits `build_pipeline()` on all of `/mnt/data/raw/train.csv`, writes `/mnt/data/models/spike/v1/model.joblib`, and prints one line `REFERENCE {...}` with the first 3 `test.csv` rows (`instances`) and their `predict_proba` (`expected`).
   - `k8s/spike-train-pod.yaml`: image `churn-base:0.1.0`, `IfNotPresent`, `restartPolicy: Never`, mounts the PVC at `/mnt/data` and a ConfigMap `spike-code` at `/code`, `PYTHONPATH=/code`. The image is not rebuilt.
   - Create the ConfigMap from `components/config.py`, `tools/spike_pipeline.py` and `tools/spike_train.py`, apply the pod, wait for `Succeeded`.
   - Save the `REFERENCE` line to `spike-reference.json`.
   - Check: `models/spike/v1/` holds only `model.joblib`, and `spike-reference.json` parses as JSON.
7. **Temporary InferenceService.**
   - `k8s/spike-inferenceservice.yaml` (data model shape).
   - Apply and wait for `Ready`.
   - Check: `kubectl get isvc churn-spike -n kubeflow` shows `READY=True`. If it never gets ready, keep the pod logs as evidence for the findings record and continue with step 9.
8. **Query the endpoint.**
   - `tools/spike_query.py` reads `spike-reference.json`, POSTs `instances` to `http://localhost:8080/v1/models/churn-spike:predict` and prints the status, the body, the shape of `predictions` and the maximum absolute difference against `expected`.
   - Run it with `kubectl port-forward -n kubeflow svc/churn-spike-predictor 8080:80` open.
   - Check: the script prints a status and a body, whatever they are. A 4xx or 5xx is a valid finding.
9. **Findings, README and cleanup.**
   - Fill the findings table in a new "Serving spike" section of `docs/runbook.md`, with the commands run, the real response shape and any error text.
   - Update the two spike risks in README §7 (probabilities, raw input) with the verified result.
   - Delete the InferenceService, the pod `spike-train` and the ConfigMap `spike-code`. Remove `models/spike/` using the `data-loader` pod of SPEC 01, then delete that pod.
   - Check: `kubectl get isvc,pod,cm -n kubeflow` lists none of the spike objects, and `ls /mnt/data` shows only `raw`.

## Acceptance criteria

### Shared config and pipeline
- [ ] `components/config.py` defines `FEATURES` with the 10 columns in the order frozen in `DECISIONS.md`, `CATEGORICAL_IDX = [1, 2]`, `NUMERIC_IDX = [0, 3, 4, 5, 6, 7, 8, 9]` and `SEED = 42`.
- [ ] `grep -rln '"CreditScore"' components tools` lists only `components/config.py`.
- [ ] `git check-ignore spike-reference.json` prints the file name.

### Baseline (E0.6)
- [ ] `PYTHONPATH=components:tools uv run python tools/baseline.py` exits 0 and prints mean and std ROC-AUC for the Dummy and the LogReg pipeline.
- [ ] The Dummy mean ROC-AUC is `0.5000`, and the LogReg mean is higher.
- [ ] The README has a "Baseline results" section whose numbers equal the script's output, rounded to 4 decimals.

### EDA notebook and summary (E0.6)
- [ ] `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb` exits 0.
- [ ] The committed notebook has at least 8 saved `image/png` outputs: target balance, numeric distributions, and churn by Geography, Gender, Age band, NumOfProducts, IsActiveMember and zero balance.
- [ ] The cell right after each chart is a markdown cell with its interpretation, in English.
- [ ] No saved output contains `CustomerId` or `Surname` (so no raw-row dump).
- [ ] `grep -c '/Users/' notebooks/eda.ipynb` returns 0.
- [ ] The notebook saves between 4 and 6 PNG files in `docs/img/eda/`, all of them charts shown in the notebook.
- [ ] `docs/eda-summary.md` exists, is in English, embeds every PNG in `docs/img/eda/`, and every embedded path resolves.
- [ ] Each finding in the summary states at least one number and names the chart it comes from.
- [ ] The summary has no raw rows, and `grep -c 'CustomerId\|Surname' docs/eda-summary.md` returns 0.
- [ ] The README links `docs/eda-summary.md` and lists `notebooks/` in §8.

### Spike: training and deployment (E0.7)
- [ ] The training pod ran image `churn-base:0.1.0`, and `models/spike/v1/` held exactly one file, `model.joblib`, before the InferenceService was created. Both facts are recorded in the runbook.
- [ ] `spike-reference.json` held 3 instances of 10 values each, with strings at positions 1 and 2. The check is recorded in the runbook.
- [ ] `k8s/spike-inferenceservice.yaml` sets runtime `kserve-sklearnserver`, `storageUri: pvc://churn-data/models/spike/v1` and `PREDICT_PROBA=true`.
- [ ] After `kubectl apply`, `kubectl get isvc churn-spike -n kubeflow` shows `READY=True`, or the pod logs that explain the failure are saved in the runbook.
- [ ] `tools/spike_query.py` prints the HTTP status, the body, the shape of `predictions` and the maximum absolute difference against `expected`.

### Spike: findings
- [ ] The "Serving spike" section of `docs/runbook.md` answers each of the four questions with exactly `yes` or `no` and a non-empty evidence column.
- [ ] The answer to the numpy/Python question is `yes` only if the recorded maximum absolute difference is below `1e-6`.
- [ ] The runbook records the real shape of `predictions`, and says which column is the churn probability.
- [ ] README §7 no longer says "unverified" for the probabilities and raw-input risks; each states the verified result.

### Cleanup and repo hygiene
- [ ] `kubectl get isvc,pod,cm -n kubeflow` lists no `churn-spike`, `spike-train` or `spike-code`, and no `data-loader` pod remains.
- [ ] `ls /mnt/data` on the PVC (checked through `data-loader` before deleting it) shows only `raw`.
- [ ] `minikube image ls | grep churn-base` shows only `churn-base:0.1.0`.
- [ ] `git ls-files | grep -c '\.csv$'` returns 0, and `git ls-files spike-reference.json` is empty.
- [ ] `git diff main -- DECISIONS.md` shows no change to the frozen contracts table.

## Decisions

- **Yes:** E0.6 and E0.7 together in SPEC 02, after SPEC 01 took the foundation. Both are local experiments that need the cluster and image from SPEC 01.
- **Yes:** a minimal `components/config.py` pulled forward from E2.1 (`FEATURES`, indices, `SEED`, paths). `FEATURES` is a frozen contract, so it should be spelled out in one place only.
- **No:** copying `FEATURES` into `tools/` scripts. The copies would drift.
- **Yes:** EDA in a notebook, with a short interpretation after each chart. The charts and their reading are the point of the exercise.
- **Yes:** commit the notebook with its outputs, with no raw-row dumps. GitHub renders it, and aggregated charts do not expose the data the repo must not publish.
- **No:** committing it without outputs. It would show nothing in the portfolio.
- **Yes:** notebook text in English, following `CLAUDE.md`.
- **Yes:** the notebook saves the 4–6 key charts with `savefig`, so re-running it regenerates them. The charts are chosen after the results are seen, and the final list is named in the summary.
- **No:** exporting every chart, or saving screenshots by hand.
- **Yes:** `docs/eda-summary.md` is written after the notebook runs, because the findings come from its results. Images live in `docs/img/eda/`, aggregated charts only, under the same data policy.
- **No:** generating the summary with `nbconvert --to markdown`, which would dump every output.
- **Yes:** baseline in `tools/baseline.py`: 5-fold stratified CV with `SEED` on all of `train.csv`, for `DummyClassifier(strategy="prior")` and a plain LogReg. The 80/20 hold-out is defined in E2.3 and stays untouched until E2.5.
- **No:** more model families or tuning in the baseline. That is E2.
- **Yes:** one throwaway `build_pipeline()` in `tools/spike_pipeline.py`, shared by baseline and spike. The final `build_preprocessor()` is E2.3.
- **Yes:** treat the 0/1 columns `HasCrCard` and `IsActiveMember` as numeric and scale them. It keeps the throwaway pipeline short, and E2.3 can revisit it.
- **Yes:** train the spike model inside a pod with `churn-base:0.1.0`, as the Fetch step will do in E3.2. This tests the real image, including its numpy and Python versions.
- **No:** training locally and copying the file to the PVC. It would not exercise the image.
- **Yes:** ship the spike scripts in a ConfigMap `spike-code` and set `PYTHONPATH=/code`. The image stays at `0.1.0` and no Docker command is needed.
- **No:** rebuilding `churn-base` with the scripts. It would bump the tag for throwaway code.
- **Yes:** `PYTHONPATH=components:tools` to import the modules, without making an installable package. It is enough for three small files.
- **Yes:** model in `models/spike/v1/`, InferenceService `churn-spike`, all deleted at the end. The future `models/churn/` stays untouched.
- **Yes:** set `runtime: kserve-sklearnserver` explicitly. The cluster has three runtimes with the `sklearn` format (`kserve-sklearnserver`, `kserve-mlserver`, `kserve-predictiveserver`), so automatic selection is ambiguous.
- **Yes:** no deployment-mode annotation. The cluster default is already `Standard`.
- **Yes:** keep testing `PREDICT_PROBA` end to end. The source of the image reads the variable, but that does not prove the answer shape.
- **Yes:** query through `tools/spike_query.py` and `kubectl port-forward`, not hand-written `curl`. A JSON with mixed strings and numbers is easy to get wrong, and the script also computes the difference against the training environment.
- **Yes:** 3 rows from `test.csv` as the reference, kept in `spike-reference.json` and git-ignored. The data policy forbids publishing rows.
- **Yes:** a `1e-6` maximum absolute difference as the threshold for "the model matches".
- **Yes:** the spec is met once each of the four questions has an answer and evidence, even if an answer is "no". The spike exists to find this out.
- **No:** requiring that the spike returns probabilities. The E0 "done when" stays open until that happens.
- **No:** a fallback (custom predictor or `FunctionTransformer`) in this spec. If the spike fails, it gets its own spec.
- **No:** the `/metrics` check on the predictor. It is E4.1, and it can wait until the final InferenceService.
- **Yes:** update the two "unverified" risks in README §7 with the verified result.

## Risks

| Risk | Mitigation |
|---|---|
| The server may turn a row of mixed strings and numbers into an all-string array, and the `ColumnTransformer` may fail or score wrongly. | This is question 2 of the spike. The response or error is recorded as evidence. If the answer is `no`, the fix (`FunctionTransformer` or custom predictor) goes in its own spec. |
| The shape of `predictions` is not what E3.4 and E4.4 expect (`[[p0, p1]]`, churn in the second column). | Step 8 records the real shape and the column that holds the churn probability in the runbook before those epics start. |
| The pod's numpy (2.4.6) and Python (3.11.18) differ in patch version from the server's (2.4.4, 3.11.16), so the pickle may not load or may score differently. | Question 3 checks load and the `1e-6` difference. If it fails, pin `numpy==2.4.4` in `pyproject.toml`, bump the image to `0.1.1` and log it in `DECISIONS.md`. |
| The executed notebook may save local paths (`/Users/...`) in warnings or tracebacks, and the repo is public. | Search the notebook for `/Users/` before committing, and re-run it with the warnings cleaned up if it appears. |
| The summary's numbers can go stale if the notebook is re-run with different data or code. | Each finding names its chart, and the summary is rechecked against the notebook after any re-run. |
| The ConfigMap `spike-code` keeps the old code if a script is edited and the pod is re-run, because pods are immutable. | Recreate the ConfigMap with `kubectl create configmap spike-code ... --dry-run=client -o yaml \| kubectl apply -f -` and delete the old pod before running it again. |
| The spike pods and the predictor add memory pressure on top of KFP and KServe, and an interrupted session can leave them running. | The cleanup step and its acceptance criteria remove every spike object. If the node is tight, apply the README memory fallback. |

## What is **not** in this spec

- Fallback for a failed spike (custom predictor or `FunctionTransformer`).
- `/metrics` check on the predictor (E4.1).
- The final `build_preprocessor()` (E2.3), tuning, more model families and drift analysis.
- MLflow, KFP pipelines, the final `churn` InferenceService, load generator and `submission.csv` (E1–E4).

Each one of those, if it lands, goes in its own spec.
