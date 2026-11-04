# SPEC 06 — Gate, Register and champion alias

> **Status:** Approved
> **Depends on:** SPEC 01, SPEC 02, SPEC 03, SPEC 04, SPEC 05
> **Backlog:** E2 (Pipeline 1 — Training), tasks E2.6, E2.7 and E2.9 of `BACKLOG.md`, plus the last part of E2.8 (the `dsl.If` Gate → Register wiring)
> **Date:** 2026-11-04
> **Objective:** Close the training pipeline with a Gate step that compares the winner's hold-out ROC-AUC against the Dummy baseline and the current champion, and a Register step that refits the winner on train + hold-out and registers it as `churn@champion` in MLflow.

## Scope

**In:**

- **E2.6 Gate component (`components/gate.py`):** compares the winner's hold-out ROC-AUC with two numbers and returns whether Register should run.
  - Inputs: `winner_holdout_auc`, `winner_family`, `winner_run_id` and `kfp_run_id`.
  - The Dummy AUC comes from the run `dummy` of the same pipeline run, found in MLflow by `kfp_run_id` and `kind=dummy`.
  - The current champion's AUC comes from the tag `holdout_roc_auc` of the version behind `models:/churn@champion`. If the alias does not exist, there is no champion and that part of the rule is skipped.
  - Rule (frozen contract): pass when `winner_holdout_auc` > Dummy AUC + 0.05 **and** `winner_holdout_auc` ≥ the champion's AUC (if one exists).
  - Outputs: `passed` (`bool`) and `reason` (`str`, with the three numbers compared). It sets the tag `gate` (`passed` or `rejected`) on the winner's parent run in MLflow.
  - A rejection is a normal result: the step succeeds with `passed=false`. The step fails only on an error: MLflow unreachable, no `dummy` run for this `kfp_run_id`, or a champion version without the tag `holdout_roc_auc`.
- **E2.7 Register component (`components/register.py`):** refits the winner on train + hold-out and registers it.
  - Inputs: the five `winner_*` outputs of Evaluate, `kfp_run_id`, `train_path` and `holdout_path`.
  - It builds the pipeline with `models.build_pipeline` from `winner_family` and `winner_params`, fits it on `train.csv` + `holdout.csv`, and logs it with `mlflow.sklearn.log_model` (signature and input example) in a new run `register` with the tags `kfp_run_id` and `kind=register`.
  - It registers the model `churn`, tags the new version with `holdout_roc_auc` (the pre-refit number from Evaluate), `winner_run_id`, `family` and `kfp_run_id`, and only then sets the alias `champion` on that version.
  - It then reloads `models:/churn@champion` and checks that `predict_proba` on a few rows returns shape `(n, 2)`.
  - Outputs: `model_version` (`str`) and `model_uri` (`str`).
  - It never reports a hold-out metric of the refit model, because that data is now part of its training set.
- **E2.8 last part:** `pipelines/training_pipeline.py` adds `dsl.If` after Evaluate: Gate runs, and Register runs only when `passed` is true. The new tasks use the same `_configure` helper (code ConfigMap, PVC, no caching, MLflow retries, `OMP_NUM_THREADS=1`). `tools/sync_code.sh` changes only if a new shared file is added.
- **Constant in `components/config.py`:** `GATE_MARGIN = 0.05`, the value of the frozen contract, kept in code and not in the tuning JSON.
- **E2.9 Run and verify:**
  - Two runs on the cluster with `n_trials: 3`: the first creates `churn` v1 with the alias `champion`, the second compares against it and, by the tie with the same seed and params, creates v2.
  - The rejection path is checked locally against a temporary MLflow store with a fake champion whose AUC is higher than the winner's.
- **Docs:** a "Gate and Register" section in `docs/runbook.md`, the ticks in `BACKLOG.md` (E2.6 to E2.9, with E2.9 noting that the long run is deferred), and the Pipeline 1 flow in README §4 brought in line with what was built (Evaluate logs no model; Register does).

**Out of scope (for future specs):**

- Pipeline 2: fetching the champion, converting it to `model.joblib`, the InferenceService, RBAC and the smoke test.
- The derived serving image for an XGBoost or LightGBM champion. Register registers the winner whatever its family.
- The full 30-trial run, deferred by the user to the end of the project. `n_trials` goes back to 30 before committing.
- A real rejection run on the cluster. The rejection path is tested locally only.
- Any change to Evaluate or to its outputs, and any change to the gate rule or the margin.
- Gate checks other than ROC-AUC (PR-AUC, recall, F1, confusion matrix).
- Deleting or archiving old model versions, and stages or transitions in the registry. Older versions stay, and the alias moves.
- Auto-retraining triggers, CI/CD, monitoring and the load generator.

## Data model

This spec adds no business data. It defines the contract of the two new components, the MLflow tags and runs they create, and the new constant. The frozen contracts of `DECISIONS.md` do not change.

### Constant added to `components/config.py`

```python
GATE_MARGIN = 0.05   # frozen contract: hold-out AUC must beat Dummy + 0.05
```

### Components (contract)

| Component | Inputs | Outputs | Fails when |
|---|---|---|---|
| `gate` | `winner_holdout_auc`, `winner_family`, `winner_run_id`, `kfp_run_id` | `passed` (`bool`), `reason` (`str`) | MLflow is unreachable, there is no `dummy` run for this `kfp_run_id`, or the champion version has no tag `holdout_roc_auc` |
| `register` | `winner_family`, `winner_params` (JSON `str`), `winner_cv_auc`, `winner_holdout_auc`, `winner_run_id`, `kfp_run_id`, `train_path`, `holdout_path` | `model_version` (`str`), `model_uri` (`str`) | MLflow is unreachable, registration fails, or the reloaded `models:/churn@champion` does not return `predict_proba` of shape `(n, 2)` |

- `passed` is `True` only if `winner_holdout_auc` > Dummy AUC + `GATE_MARGIN` **and** (there is no champion **or** `winner_holdout_auc` ≥ the champion's `holdout_roc_auc`).
- `reason` is one line with the three numbers, for example `winner 0.8812 > dummy 0.5000 + 0.05 and >= champion v1 0.8812`. With no champion it says so.
- Each component lives in its own file under `components/` and is a `@dsl.component(base_image="churn-base:0.3.0")` function. Imports go inside the body.
- `kfp_run_id` receives `dsl.PIPELINE_JOB_ID_PLACEHOLDER`.
- Every task goes through `_configure`: `PYTHONPATH=/code`, caching off, `MLFLOW_HTTP_REQUEST_MAX_RETRIES=10`, `OMP_NUM_THREADS=1`.
- Resources: `gate` asks 100m CPU and 256Mi, with a 1Gi limit. `register` asks 500m and 512Mi, with a 1.5Gi limit, like `tune`, because it fits the winner once on 18,000 rows.

### Pipeline `churn-training`

```text
ingest → preprocess → load_config → ParallelFor(family, parallelism=2): tune(family)
       → evaluate(Collected(tune results))
       → gate(evaluate outputs)
       → dsl.If(gate.passed): register(evaluate outputs)
```

### MLflow layout additions (experiment `churn-training`)

| Object | Name | Tags | Contents |
|---|---|---|---|
| Tag on the parent run of the winner | — | `gate` = `passed` or `rejected` | set by Gate |
| Run, one per Register | `register` | `kfp_run_id`, `kind=register`, `family` | the logged model (artifact path `model`, signature and input example); params `family` and `best.<param>`; metric `winner_holdout_auc` copied from Evaluate (the pre-refit number) |
| Registered model | `churn` | — | one new version per Register |
| Model version tags | — | `holdout_roc_auc`, `winner_run_id`, `family`, `kfp_run_id` | `holdout_roc_auc` is what the next Gate reads; set **before** the alias moves |
| Alias | `champion` | — | moved to the newest version by Register, after its tags are set |

- The signature takes the raw `FEATURES` columns as input and `predict_proba` as output, which is what Pipeline 2 will serve. The input example is five rows of `train.csv` in `FEATURES` order.
- The refit model is fitted on `train.csv` + `holdout.csv`, so no hold-out metric is computed or logged for it.
- Gate finds the Dummy with `kind = 'dummy' and tags.kfp_run_id = '<kfp_run_id>'` and reads `holdout_roc_auc`. It finds the champion with `MlflowClient.get_model_version_by_alias("churn", "champion")`, and a missing alias means "no champion".
- Old versions are never deleted: the alias moves to the new version and the rest stay in the registry.

### Cluster objects

| Object | Namespace | Notes |
|---|---|---|
| ConfigMap `churn-code` | `kubeflow` | Unchanged unless a new shared file is added |
| Pipeline `churn-training` | `kubeflow` | Seven kinds of step: ingest, preprocess, load_config, tune, evaluate, gate and register |
| Registered model `churn` | MLflow in `mlflow` | Created by the first Register |

## Implementation plan

1. **Constant and the pure gate rule.**
   - Add `GATE_MARGIN = 0.05` to `components/config.py` and `gate_decision(winner_auc, dummy_auc, champion_auc)` to `components/models.py`. It returns `(passed, reason)`, and `champion_auc` is `None` when there is no champion.
   - Check, locally, with plain numbers: no champion and a winner above Dummy + 0.05 passes. A winner at exactly Dummy + 0.05 is rejected (the comparison is strict). A winner above the margin but below the champion is rejected. A winner equal to the champion passes. The `reason` of each case names the numbers compared.
2. **Gate component (E2.6).**
   - Write `components/gate.py`: find the `dummy` run by `kfp_run_id` and `kind`, read the champion through the alias (a missing alias means no champion), call `gate_decision`, set the tag `gate` on the winner's parent run, and return `passed` and `reason`.
   - Check, locally, against a temporary MLflow store (a SQLite file in a temporary folder outside the repo, because the registry needs a database-backed store) with a fake `dummy` run, a fake winner parent run and a tiny `LogisticRegression` registered as `churn@champion`:
     - No alias: `passed` is true and the winner's parent run has `gate=passed`.
     - Champion tag lower than the winner: `passed` is true.
     - Champion tag higher than the winner (the fake champion of the rejection path): `passed` is false, the step does not raise and the parent run has `gate=rejected`.
     - Champion version without the tag `holdout_roc_auc`: the step raises with a message that names the version.
     - No `dummy` run for the `kfp_run_id`: the step raises.
3. **Register component (E2.7).**
   - Write `components/register.py`: read `train.csv` and `holdout.csv`, build the pipeline with `models.build_pipeline`, fit it on both, log it in a run `register` with signature and input example, register `churn`, set the four version tags, **then** set the alias `champion`, reload `models:/churn@champion`, check the shape of `predict_proba`, and return `model_version` and `model_uri`.
   - Check, locally, with a split made by `preprocess` in a temporary folder and the temporary store of step 2:
     - The first call returns version `1`, the alias `champion` points to it, and the version has the four tags.
     - The calls to `MlflowClient` happen in this order: the four `set_model_version_tag` calls, then `set_registered_model_alias`.
     - A second call returns version `2`, the alias moves to `2`, and version `1` still exists.
     - The run `register` has `kind=register`, the artifact `model` and no `holdout_*` metric.
     - `grep -c roc_auc_score components/register.py` prints 0.
4. **Pipeline wiring (E2.8, last part).**
   - Extend `pipelines/training_pipeline.py` as in the data model: `gate` after `evaluate`, and `register` inside `dsl.If(gate_task.outputs["passed"] == True)`. Both go through `_configure`, with the resources of the data model.
   - Check: the pipeline compiles. The compiled YAML has a condition on the output of the gate task, has `register` inside it, and references `churn-base:0.3.0` and not `0.2.0`. `tools/sync_code.sh` still creates a ConfigMap with four keys. If `dsl.If` cannot compare the `bool` output as written, stop and bring the options to the user.
5. **First run on the cluster, with 3 trials.**
   - Set `n_trials` to 3 in `tuning_config.json`, run `tools/sync_code.sh`, open `kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888` and `kubectl port-forward -n mlflow svc/mlflow 5001:5000`, and run `tools/submit_pipeline.py --wait`.
   - Check: the run is `Succeeded`. The winner's parent run has `gate=passed`. MLflow shows `churn` version 1 with the alias `champion` and its four tags, and the run `register` with the model artifact. The logs show no `database is locked`. Note the time of the run and the size of the model artifact.
   - If a step fails on MLflow, memory or the image, save the pod logs and events as evidence and stop.
6. **Second run on the cluster, against the existing champion.**
   - Run `tools/submit_pipeline.py --wait` again with the same 3 trials. With the same seed and params, the winner ties with version 1.
   - Check: the run is `Succeeded`. The `reason` of the gate in the log names champion version 1. `churn` has version 2, the alias `champion` points to it, and version 1 still exists. Note the time of the run.
7. **Docs and hygiene.**
   - Write the "Gate and Register" section of `docs/runbook.md` with the commands actually run, the observed results, the times of both runs and the gotchas. Tick E2.6 to E2.9 in `BACKLOG.md` (E2.9 notes that the long run is still deferred) and update the Pipeline 1 flow in README §4.
   - Put `n_trials` back to 30 in `tuning_config.json`.
   - Check: every command in the runbook section appears in the steps above, and `git diff main -- components/tuning_config.json` is empty.

## Acceptance criteria

### Config and gate rule
- [ ] `components/config.py` defines `GATE_MARGIN = 0.05`, and `FEATURES`, `SEED`, `MODEL_NAME` and `CHAMPION_ALIAS` are unchanged.
- [ ] `gate_decision` passes with no champion when the winner is above Dummy + 0.05, rejects a winner at exactly Dummy + 0.05, rejects a winner below the champion, and passes a winner equal to the champion. The `reason` of each case names the numbers compared.

### Gate (local, temporary MLflow store)
- [ ] With no alias `champion`, `gate` returns `passed=True` and the winner's parent run has `gate=passed`.
- [ ] With a champion whose `holdout_roc_auc` tag is lower than the winner's, `gate` returns `passed=True`.
- [ ] With a champion whose `holdout_roc_auc` tag is higher than the winner's, `gate` returns `passed=False` without raising, and the winner's parent run has `gate=rejected`.
- [ ] With a champion version that has no tag `holdout_roc_auc`, `gate` raises with a message that names the version.
- [ ] With no `dummy` run for the `kfp_run_id`, `gate` raises.

### Register (local, temporary MLflow store)
- [ ] The first call returns version `1`, the alias `champion` points to it, and the version has the tags `holdout_roc_auc`, `winner_run_id`, `family` and `kfp_run_id`.
- [ ] The alias `champion` is moved after the version has its four tags: the four `set_model_version_tag` calls come before `set_registered_model_alias`.
- [ ] The tag `holdout_roc_auc` of the version equals the `winner_holdout_auc` that was passed in (the pre-refit number).
- [ ] A second call returns version `2`, the alias moves to `2`, and version `1` still exists.
- [ ] The run `register` has `kind=register`, the artifact `model` with signature and input example, and no `holdout_*` metric.
- [ ] `grep -c roc_auc_score components/register.py` prints 0: Register never scores the refit model.
- [ ] The reloaded `models:/churn@champion` returns `predict_proba` of shape `(n, 2)` on raw rows in `FEATURES` order.

### Pipeline on the cluster
- [ ] The compiled pipeline YAML has a condition on the output of the gate task, has `register` inside it, and references `churn-base:0.3.0` and not `0.2.0`.
- [ ] `kubectl get cm churn-code -n kubeflow` still holds the four keys `config.py`, `preprocessing.py`, `models.py` and `tuning_config.json`.
- [ ] The first run with 3 trials finishes `Succeeded`. For that run (filtering by `kfp_run_id`) the winner's parent run has `gate=passed`, `churn` has version 1 with the alias `champion` and its four tags, and the run `register` has the model artifact.
- [ ] The second run with 3 trials finishes `Succeeded`. The gate log names champion version 1, `churn` has version 2 with the alias `champion`, and version 1 still exists.
- [ ] The logs of both runs show no `database is locked`.
- [ ] Every run of both pipeline runs, including `register`, has the tag `kfp_run_id`.
- [ ] `kubectl get pods -A --no-headers | awk '$4!="Running" && $4!="Completed"'` prints nothing, and no temporary pod is left in `kubeflow`.

### Docs and hygiene
- [ ] `docs/runbook.md` has a "Gate and Register" section with the commands run, the observed results, the times of both runs and the gotchas.
- [ ] `BACKLOG.md` has E2.6 to E2.9 ticked, and E2.9 says the long run is still deferred.
- [ ] README §4 says that Evaluate logs no model and that Register logs and registers it.
- [ ] `git diff main -- DECISIONS.md` shows no change to the frozen contracts table.
- [ ] `git diff main -- components/tuning_config.json` is empty (`n_trials` is 30), and `git ls-files | grep -c '\.csv$'` returns 0.

## Decisions

- **Yes:** one spec for Gate, Register, the `dsl.If` wiring and the second run. Register needs the gate's answer to run, and together they leave a run that ends in `churn@champion`.
- **No:** two specs, Gate first and Register after. The first run would show a gate with nothing to register and no champion to compare against.
- **Yes:** the gate reads the champion's AUC from the tag `holdout_roc_auc` on its model version, which Register writes from the pre-refit number of Evaluate. The champion is refit on train + hold-out, so its own hold-out AUC is no longer clean, and the tag keeps the number next to the version it describes.
- **No:** following `winner_run_id` to the parent run and reading the metric there. It is one more hop, and it breaks if the run is deleted.
- **Yes:** the gate finds the Dummy AUC in the run `dummy` of the same pipeline run, by `kfp_run_id` and `kind=dummy`. Evaluate, already merged, stays untouched.
- **No:** a new output `dummy_holdout_auc` in Evaluate. It is simpler, but it reopens a spec that is `Implemented`.
- **Yes:** a rejection is a normal result: the gate step succeeds with `passed=false`, `dsl.If` skips Register and the run ends `Succeeded`. A model that is not better is not an error.
- **No:** failing the step on rejection. A red run would hide real failures behind an expected result.
- **Yes:** the gate fails the step when the alias `champion` exists but its version has no tag `holdout_roc_auc`, and when there is no `dummy` run. Passing in silence would let a model through without the comparison the contract asks for.
- **Yes:** the gate sets the tag `gate` (`passed` or `rejected`) on the winner's parent run, so the decision shows in the MLflow UI without opening KFP.
- **Yes:** `GATE_MARGIN = 0.05` lives in `config.py`, not in the tuning JSON. It is a frozen contract, like `SEED` and the split.
- **Yes:** the rule is strict on the margin (`>` Dummy + 0.05) and not strict on the champion (`≥`), as the frozen contract says. A tie with the champion passes, so a retrain with the same result replaces it.
- **Yes:** the pure rule lives in `models.gate_decision`, so the five cases can be checked with plain numbers before any MLflow is involved.
- **Yes:** Register refits the winner on train + hold-out and logs the model in a new run `register`, with the tags `kfp_run_id` and `kind=register`. The family's parent run keeps only the tuning and evaluation results, and the model gets its own run.
- **No:** logging the model in the winner's parent run. That run is closed by Tune and reopened by Evaluate, and it would mix the pre-refit numbers with a model fitted on other data.
- **Yes:** Register sets the four tags of the version before it moves the alias. A failure between the calls can then never leave a champion without the tag that the next gate reads.
- **No:** moving the alias first. A failure after it would leave `churn@champion` pointing to a version the gate cannot compare against, and every later run would fail on it.
- **Yes:** Register never computes a metric on the refit model. The hold-out is part of its training set, so any score would be inflated. The reported number is the pre-refit one from Evaluate, copied as a metric of the `register` run and as a tag of the version.
- **Yes:** the signature takes the raw `FEATURES` columns as input and `predict_proba` as output, with five rows of `train.csv` as the input example. It is the same shape that Pipeline 2 serves.
- **Yes:** Register reloads `models:/churn@champion` and checks the shape of `predict_proba`. It costs seconds and proves the registry round trip that Pipeline 2 depends on.
- **No:** skipping the reload and leaving the first load to Pipeline 2. A broken registration would show up two specs later and far from its cause.
- **Yes:** old versions stay in the registry and only the alias moves. It keeps the lineage, and the registry is small.
- **No:** deleting or archiving old versions, or using stages. Stages are deprecated in MLflow 3, and the README chose aliases.
- **Yes:** Register registers the winner whatever its family. A champion of XGBoost or LightGBM cannot be served by `sklearnserver` yet, and the derived image goes in the spec of Pipeline 2, as decided in SPEC 05.
- **No:** a `registrable` flag per family. It was already discarded in SPEC 05: the winner would stop being the best by CV.
- **Yes:** the rejection path is checked locally, with a fake champion whose AUC is higher, in a temporary MLflow store. It is a SQLite file and not a file store as in SPEC 05, because the registry with aliases needs a database-backed store.
- **No:** a real rejection run on the cluster. It needs a manual edit of a tag and a third run. The local check covers the logic, and the two cluster runs cover the integration.
- **Yes:** two cluster runs with 3 trials: the first creates version 1, and the second ties with it (same seed and params) and creates version 2. It checks both the empty-registry case and the comparison against an existing champion.
- **Yes:** the full 30-trial run stays deferred to the end of the project, as decided in SPEC 05. `n_trials` goes back to 30 before committing.
- **Yes:** the pipeline keeps using `_configure` for the two new tasks. The code ConfigMap, the PVC, caching off and the MLflow retries then apply to them with no special case.
- **No:** a change to the frozen contracts. The gate rule, the registry names and the split are used as written, so `DECISIONS.md` needs no new line.

## Risks

| Risk | Mitigation |
|---|---|
| The second run does not tie with version 1: a small float difference between pods makes its hold-out AUC slightly lower, and the gate rejects it. Version 2 is not created and the criterion is not met. | Same seed, `n_jobs: 1` and `OMP_NUM_THREADS=1` in both runs, so a tie is the expected result. If it still rejects, the `reason` of the gate shows both numbers. Stop and bring them to the user. Do not edit tags by hand to force version 2. |
| `dsl.If` cannot compare the `bool` output of the gate as written in KFP 2.15. | Step 4 compiles and inspects the YAML before the cluster. If it fails, stop and bring the options to the user (for example a `str` output compared with `"true"`). |
| Register fails between its calls and leaves version N with the alias `champion` but without the tag `holdout_roc_auc`. The next gate would then fail on it by design. | Register sets the version tags **before** moving the alias, so a failure can never leave a champion without its tag. Step 3 checks the order of the calls. |
| `log_model` fails in the pod, or the model does not reload there (missing library, or a version different from training). | The reload of `models:/churn@champion` inside Register fails the step in the same pod, with the same image that Pipeline 2 will use. Step 5 shows the traceback in the log of the step. |
| Each registered model is a full fitted `Pipeline` on the MLflow PVC (2Gi). A Random Forest of 400 trees can weigh tens of MB, and every run adds another. | Two runs in this spec. Step 5 notes the size of the artifact in the runbook. If it grows too much, raising the PVC is a decision for a later spec. |
| The temporary SQLite store of the local checks behaves differently from the MLflow server of the cluster (same client version, different backend and artifact path). | The local checks cover the logic only. The first real contact with the server is step 5, and the two cluster runs are the integration test. |
| KFP runs fail in their first minutes (drivers or launchers cannot reach `metadata-grpc` or MySQL). The cause was not found. After a `minikube stop` the port-forwards die and the CPU quota may be lower than 8. | Resubmit. A Python traceback in a pod is a real failure and not this flake. Before step 5, check `cpu.max` of the node and reopen the port-forwards. Delete the Error pods that failed attempts leave. The checks filter by `kfp_run_id`, so partial runs do not count. |

## What is **not** in this spec

- Pipeline 2: fetching the champion, `model.joblib`, the InferenceService, RBAC and the smoke test.
- The derived serving image for an XGBoost or LightGBM champion.
- The full 30-trial run (deferred to the end of the project).
- A real rejection run on the cluster.
- Any change to Evaluate, to its outputs, to the gate rule or to the margin.
- Gate checks other than ROC-AUC.
- Deleting or archiving old model versions, and registry stages.
- Auto-retraining triggers, CI/CD, monitoring and the load generator.

Each one of those, if it lands, goes in its own spec.
