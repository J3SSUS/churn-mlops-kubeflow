# SPEC 05 — Config-driven tuning and evaluation (Tune per family, Evaluate)

> **Status:** Approved
> **Depends on:** SPEC 01, SPEC 02, SPEC 03, SPEC 04
> **Backlog:** E2 (Pipeline 1 — Training), tasks E2.4 and E2.5 of `BACKLOG.md`, plus the second part of E2.8 (`ParallelFor` wiring) and the `churn-base:0.3.0` image
> **Date:** 2026-11-02
> **Objective:** Extend the training pipeline with a Tune step per model family, driven by `tuning_config.json` (Optuna, nested MLflow runs, `ParallelFor`), and an Evaluate step that scores the best config of each family on the hold-out against a Dummy baseline.

## Why this spec exists

SPEC 04 left a green run of Ingest → Preprocess and the split on the PVC. This spec adds the part that trains: one Tune step per model family and an Evaluate step. It is also the first spec where the pipeline pods call MLflow, so the risk moves from the cluster plumbing to the tracking server.

Four things are not obvious:

- **The families are not in the code.** They live in `components/tuning_config.json`: the class of each family, its fixed parameters, its search ranges and the number of trials. A small `load_config` step reads and validates the file inside the cluster and gives the list of families to `ParallelFor`. Adding a hyperparameter or a family means editing the JSON and submitting again, with no recompile and no change to Python.
- **The hold-out stays clean.** The winner is the family with the best CV ROC-AUC from Tune. Evaluate scores every family on the hold-out only to report it, so no choice is made looking at those 3,000 rows. The gate of SPEC 06 will read a number that did not take part in any selection.
- **XGBoost and LightGBM need a new image.** `churn-base:0.3.0` adds both libraries and `libgomp1`, which LightGBM needs and `python:3.11-slim` does not carry. The KServe `sklearnserver` image has neither library, so a champion of those families could not be served yet. That is solved in the spec of Pipeline 2, and it is recorded here as a risk.
- **No model is logged in MLflow here.** Tune logs params and CV metrics, and Evaluate logs hold-out metrics. Only Register (SPEC 06) logs and registers the final model, which keeps the 2Gi artifact PVC small.

SPEC 06 follows with the Gate, Register, the alias `champion` and the second run.

## Scope

**In:**

- **Image `churn-base:0.3.0`:** `xgboost` and `lightgbm` added to the runtime dependencies with pinned versions, `libgomp1` installed in the `Dockerfile`, and the image rebuilt by the user. Every component of the pipeline moves to `0.3.0`. `0.2.0` stays in minikube.
- **Tuning config (`components/tuning_config.json`):** global `n_trials` and `cv_folds`, and a block per family with its `class`, `fixed` parameters, `search` ranges and an optional `n_trials` of its own. It ships with five families: `logreg`, `random_forest`, `hist_gradient_boosting`, `xgboost` and `lightgbm`, with 30 trials each.
- **Shared module `components/models.py`:** validates the config, suggests the parameters of a trial from the `search` block, and builds the sklearn `Pipeline` (`build_preprocessor()` plus the model) for a family. Tune and Evaluate use it now, and Register will use it in SPEC 06.
- **E2.x `load_config` component:** reads the JSON from `/code`, validates it and fails the step with a clear message on any error. It returns the list of families for `ParallelFor`.
  - The class must start with `sklearn.`, `xgboost.` or `lightgbm.` and must have `predict_proba`.
  - Every `fixed` and `search` parameter must exist in that class.
  - Every range must have `low` < `high`, and a `categorical` needs at least one choice.
- **E2.4 Tune component:** one family per call. An Optuna study (TPE sampler with `SEED`, `MedianPruner` reporting after each fold) maximizes the mean ROC-AUC of a stratified CV with `cv_folds` folds, fitting the preprocessor inside each fold.
  - MLflow: a parent run per family and a nested run per trial, with params and the CV metrics. The effective `tuning_config.json` is logged as an artifact of the parent run. Every run carries a tag with the KFP run id.
  - It returns the family, the best params, the best CV AUC and the parent `run_id`.
- **E2.5 Evaluate component:** receives the results of all the families. For each one it refits the best config on all of `train.csv`, with the preprocessor fitted on it, and never uses the CV fold models. It then scores `holdout.csv`: ROC-AUC, PR-AUC, recall, F1 and the confusion matrix. It does the same with `DummyClassifier(strategy="prior")`.
  - MLflow: the hold-out metrics go to the parent run of each family (`holdout_roc_auc`, `holdout_pr_auc`, `holdout_recall`, `holdout_f1` and the matrix), and the Dummy gets its own run `dummy`.
  - It returns the winner: the family with the best CV AUC, its params, its CV AUC, its hold-out AUC and its parent `run_id`.
- **E2.8 second part:** `pipelines/training_pipeline.py` grows to Ingest → Preprocess → Load config → `ParallelFor` over the families (`parallelism=2`) running Tune → Evaluate with the collected results. `tools/sync_code.sh` adds `models.py` and `tuning_config.json` to the ConfigMap `churn-code`.
- **Run and check:** a first run with `n_trials: 3` to check MLflow from the pods, `ParallelFor` and SQLite, and then a full run with the defaults. MLflow shows five parent runs with nested trials, the hold-out metrics and the `dummy` run.
- **Docs:** a section in `docs/runbook.md`, two lines in `DECISIONS.md` (the tag `0.3.0` and the `xgboost` and `lightgbm` pins), and the `churn-base` row of the README.

**Out of scope (for future specs):**

- Gate, Register, the alias `champion` and the second run (SPEC 06).
- Logging or registering any model in MLflow (Register, SPEC 06).
- Serving a champion of XGBoost or LightGBM: the derived `sklearnserver` image goes in the spec of Pipeline 2.
- Choosing the winner by hold-out AUC. The hold-out only reports.
- Anything in the JSON about `SEED`, the split, `FEATURES`, the sampler or the pruner. They stay in the code, because the first three are frozen contracts.
- A different preprocessor per family. All of them use `build_preprocessor()`.
- Libraries other than scikit-learn, XGBoost and LightGBM, and any change of the base image beyond `libgomp1`.
- A shared Optuna database (RDB), Katib and early stopping inside XGBoost or LightGBM.
- Removing `churn-base:0.2.0` from minikube.
- Pipeline 2, monitoring and the load generator.

## Data model

This spec adds no business data. It defines the tuning config, the contract of each component, the layout of the MLflow runs and the new cluster objects. The frozen contracts of `DECISIONS.md` do not change.

### Constant added to `components/config.py`

```python
TUNING_CONFIG_PATH = "/code/tuning_config.json"   # the ConfigMap mount; tests point it elsewhere
```

### `components/tuning_config.json` (default)

```json
{
  "n_trials": 30,
  "cv_folds": 5,
  "families": {
    "logreg": {
      "class": "sklearn.linear_model.LogisticRegression",
      "fixed": {"max_iter": 1000},
      "search": {"C": {"type": "float", "low": 0.001, "high": 100, "log": true}}
    },
    "random_forest": {
      "class": "sklearn.ensemble.RandomForestClassifier",
      "fixed": {"n_jobs": 1},
      "search": {
        "n_estimators": {"type": "int", "low": 100, "high": 400},
        "max_depth": {"type": "int", "low": 3, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]}
      }
    },
    "hist_gradient_boosting": {
      "class": "sklearn.ensemble.HistGradientBoostingClassifier",
      "fixed": {},
      "search": {
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": true},
        "max_iter": {"type": "int", "low": 100, "high": 400},
        "max_leaf_nodes": {"type": "int", "low": 8, "high": 64},
        "l2_regularization": {"type": "float", "low": 0.001, "high": 10, "log": true}
      }
    },
    "xgboost": {
      "class": "xgboost.XGBClassifier",
      "fixed": {"n_jobs": 1, "tree_method": "hist", "eval_metric": "logloss"},
      "search": {
        "n_estimators": {"type": "int", "low": 100, "high": 400},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": true},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0}
      }
    },
    "lightgbm": {
      "class": "lightgbm.LGBMClassifier",
      "fixed": {"n_jobs": 1, "verbose": -1, "subsample_freq": 1},
      "search": {
        "n_estimators": {"type": "int", "low": 100, "high": 400},
        "num_leaves": {"type": "int", "low": 8, "high": 64},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": true},
        "min_child_samples": {"type": "int", "low": 5, "high": 50},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0}
      }
    }
  }
}
```

Rules of the file:

- `n_trials` and `cv_folds` are global. A family can set its own `n_trials`.
- A `search` entry is `int` or `float` (`low`, `high`, optional `log`) or `categorical` (`choices`).
- A parameter cannot be in both `fixed` and `search`.
- `models.py` sets `random_state=SEED` when the class accepts it and the config does not set it. The file never holds `SEED`, the split or `FEATURES`.
- `subsample_freq` is fixed to 1 for LightGBM because, with its default of 0, LightGBM ignores `subsample`.

### Components (contract)

| Component | Inputs | Output | Fails when |
|---|---|---|---|
| `load_config` | none (reads `TUNING_CONFIG_PATH`) | the list of family names (`List[str]`) | the config is invalid (see rules above) or has no families |
| `tune` | `family`, `kfp_run_id`, the processed `train_path` | `result_json` (`str`) with `family`, `best_params`, `best_cv_auc`, `run_id` | the config is invalid, MLflow is unreachable, or no trial completes |
| `evaluate` | `tune_results` (`List[str]`, the collected `result_json`), `kfp_run_id`, `train_path`, `holdout_path` | `winner_family`, `winner_params` (`str`, JSON), `winner_cv_auc`, `winner_holdout_auc`, `winner_run_id` | a result is missing or MLflow is unreachable |

- The winner is the family with the highest `best_cv_auc`. A tie goes to the family that comes first in the config.
- Each component lives in its own file under `components/` and is a `@dsl.component(base_image="churn-base:0.3.0")` function. Imports go inside the body.
- `kfp_run_id` receives `dsl.PIPELINE_JOB_ID_PLACEHOLDER`.
- Every task sets `PYTHONPATH=/code`, caching off and `MLFLOW_HTTP_REQUEST_MAX_RETRIES=10`.
- Resources: `tune` asks 500m CPU and 512Mi, with a 1.5Gi limit. `load_config` and `evaluate` ask 100m and 256Mi, with a 1Gi limit.

### Pipeline `churn-training`

```text
ingest → preprocess → load_config → ParallelFor(family, parallelism=2): tune(family) → evaluate(Collected(tune results))
```

### MLflow layout (experiment `churn-training`)

| Run | Name | Tags | Params and metrics |
|---|---|---|---|
| Parent, one per family | `<family>` | `kfp_run_id`, `family`, `kind=family` | params `class`, `n_trials`, `cv_folds`, `best.<param>`; metrics `best_cv_roc_auc`, `n_complete`, `n_pruned`; artifact `tuning_config.json`; later, from Evaluate: `holdout_roc_auc`, `holdout_pr_auc`, `holdout_recall`, `holdout_f1`, `holdout_tn`, `holdout_fp`, `holdout_fn`, `holdout_tp` |
| Nested, one per trial | `<family>-trial-<n>` | `kfp_run_id`, `family`, `kind=trial`, `trial_state=COMPLETE` or `PRUNED` | the suggested params; metrics `cv_roc_auc` (mean of the folds seen), `cv_roc_auc_std`, `folds_done` |
| Baseline | `dummy` | `kfp_run_id`, `kind=dummy` | the same `holdout_*` metrics |

- Evaluate reopens each parent run with the `run_id` that Tune returned and logs the hold-out metrics there.
- The labels behind recall, F1 and the confusion matrix use a threshold of 0.5. ROC-AUC and PR-AUC use the probabilities.
- No model is logged. The only artifact is `tuning_config.json`.
- The CV is `StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)`, and the sampler is `TPESampler(seed=SEED)`.

### Cluster objects

| Object | Namespace | Notes |
|---|---|---|
| Image `churn-base:0.3.0` | — | `0.2.0` plus `xgboost`, `lightgbm` and `libgomp1`, built by the user |
| ConfigMap `churn-code` | `kubeflow` | Four keys now: `config.py`, `preprocessing.py`, `models.py` and `tuning_config.json` |
| Pipeline `churn-training` | `kubeflow` | Seven kinds of step as above, with one `tune` task per family |

## Implementation plan

1. **Decision lines, dependencies and Dockerfile (image prerequisites).**
   - Add two lines to the `DECISIONS.md` log: the image tag `churn-base:0.3.0` (`0.2.0` plus `xgboost`, `lightgbm` and `libgomp1`), and the `xgboost` and `lightgbm` pins, noting that the KServe `sklearnserver` image has neither library and that the Pipeline 2 spec must build a derived image with the same versions.
   - In `pyproject.toml`, add `xgboost` and `lightgbm` to the runtime dependencies with the exact versions that `uv` resolves. Run `uv lock` and regenerate `docker/churn-base/requirements.txt` with the `uv export` command in its header.
   - In `docker/churn-base/Dockerfile`, install `libgomp1` with `apt-get` (and clean the apt lists in the same layer).
   - Check: `git diff docker/churn-base/requirements.txt` adds `xgboost`, `lightgbm` and their dependencies only, `scikit-learn` is still `1.5.2` and `numpy` is still `2.4.6`, and `uv run python -c "import xgboost, lightgbm"` exits 0.
2. **Build the image `churn-base:0.3.0` (user runs it).**
   - The user runs `minikube image build` with the new tag, with the same command as in the runbook.
   - Check: `minikube image ls | grep churn-base` shows `0.1.0`, `0.2.0` and `0.3.0`. A temporary pod with `0.3.0` prints the versions of `kfp` (2.15.0), `scikit-learn` (1.5.2), `mlflow` (3.17.2), `xgboost` and `lightgbm`, and fits a tiny `XGBClassifier` and a tiny `LGBMClassifier` (this proves `libgomp1`). Delete the pod afterwards.
3. **Tuning config and its validation.**
   - Add `TUNING_CONFIG_PATH` to `components/config.py`, write `components/tuning_config.json` as in the data model, and start `components/models.py` with `load_tuning_config(path)` and `validate_config(cfg)`.
   - Check, locally: the default config passes. Each of these raises with a message that names the problem: a class outside `sklearn.`, `xgboost.` and `lightgbm.`, a class without `predict_proba`, a parameter the class does not have, `low` >= `high`, a parameter in both `fixed` and `search`, an empty `choices`, and a config with no families. The bad copies live outside the repo.
4. **Trial parameters and model builder.**
   - Add `suggest_params(trial, search)` and `build_pipeline(family_cfg, params)` to `components/models.py`. The pipeline is `build_preprocessor()` plus the model, and `random_state=SEED` is set when the class accepts it and the config does not.
   - Check, locally on a 2,000-row sample of `data/train.csv`: for each of the five families, a fixed Optuna trial gives params, `build_pipeline` fits, and `predict_proba` returns shape `(n, 2)`. The model has the `random_state` that `SEED` sets.
5. **`load_config` component.**
   - Write `components/load_config.py`.
   - Check, locally with `TUNING_CONFIG_PATH` pointed at the repo file: it returns the five family names in the file order. On a temporary invalid copy it raises.
6. **Tune component (E2.4).**
   - Write `components/tune.py`: the Optuna study (TPE with `SEED`, `MedianPruner`, stratified CV that fits the pipeline inside each fold), the parent and nested MLflow runs with their tags, the `tuning_config.json` artifact, and the `result_json` output.
   - Check, locally against a temporary MLflow file store (`MLFLOW_TRACKING_URI` pointed at it), with `n_trials` 3 and `cv_folds` 3 for `logreg` and one boosting family: `result_json` parses and holds the four fields. The store has one parent run per family with `kind=family`, the artifact `tuning_config.json`, and three nested runs with `kind=trial`. `best_cv_auc` for `logreg` is at least 0.87. With 12 trials of `logreg`, `n_complete + n_pruned` equals 12.
7. **Evaluate component (E2.5).**
   - Write `components/evaluate.py`.
   - Check, locally with the file store of step 6 and a split made by `preprocess` in a temporary folder: it reopens each parent run and logs the eight `holdout_*` metrics, and it creates the `dummy` run with `holdout_roc_auc` of 0.5. `holdout_tn + holdout_fp + holdout_fn + holdout_tp` equals 3,000. The winner is the family with the highest `best_cv_auc`. No run has a model artifact.
8. **Pipeline wiring (E2.8, second part).**
   - Extend `pipelines/training_pipeline.py` as in the data model, with `ParallelFor(parallelism=2)` and `dsl.Collected`. Move `ingest` and `preprocess` to `churn-base:0.3.0`, set the resources and the `MLFLOW_HTTP_REQUEST_MAX_RETRIES` variable, and add `models.py` and `tuning_config.json` to `tools/sync_code.sh`.
   - Check: the pipeline compiles. The YAML references `churn-base:0.3.0` and not `0.2.0`, has a `ParallelFor` with `parallelism: 2`, and the sync script creates a ConfigMap with four keys. If `dsl.Collected` cannot carry the results as described, stop and bring the options to the user.
9. **First run on the cluster, with 3 trials.**
   - Set `n_trials` to 3 in `tuning_config.json`, open `kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888` and `kubectl port-forward -n mlflow svc/mlflow 5001:5000`, and run `tools/submit_pipeline.py --wait`.
   - Check: the run is `Succeeded`. MLflow shows five parent runs with three nested runs each, the `holdout_*` metrics in each parent and the `dummy` run. The logs show no `database is locked`. Note the time of the run.
   - If a step fails on MLflow, memory or the image, save the pod logs and events as evidence and stop. Put `n_trials` back to 30 before the next step.
10. **Full run, with the defaults.**
    - Run `tools/submit_pipeline.py --wait` with the 30 trials of the file.
    - Check: the run is `Succeeded`. Each family has 30 nested runs and `n_complete + n_pruned` equals 30. Every family has `holdout_roc_auc` above the `dummy` run. The winner is the family with the highest `best_cv_auc`. Note the time of the run and any pod restart.
11. **Docs.**
    - Write the "Config-driven tuning and evaluation" section of `docs/runbook.md` with the commands actually run, the observed results, the times and the gotchas. In the README, update the `churn-base` row to `0.3.0`, the `Models` and `Tuning` rows of §5 (five families by default, set in `tuning_config.json`) and add the serving risk of XGBoost and LightGBM to §7.
    - Check: every command in the runbook section appears in the steps above.

## Acceptance criteria

### Image and dependencies
- [ ] `pyproject.toml` has `xgboost==` and `lightgbm==` (exact versions) in the runtime dependencies, and `uv run python -c "import xgboost, lightgbm"` exits 0.
- [ ] `git diff main -- docker/churn-base/requirements.txt` adds `xgboost`, `lightgbm` and their dependencies, `scikit-learn` is still `1.5.2` and `numpy` is still `2.4.6`.
- [ ] `docker/churn-base/Dockerfile` installs `libgomp1`.
- [ ] `minikube image ls | grep churn-base` lists `0.1.0`, `0.2.0` and `0.3.0`.
- [ ] A pod with `churn-base:0.3.0` prints the versions of `kfp` (2.15.0), `scikit-learn` (1.5.2), `mlflow` (3.17.2), `xgboost` and `lightgbm`, and fits a tiny `XGBClassifier` and a tiny `LGBMClassifier` without error.

### Config and models
- [ ] `components/config.py` defines `TUNING_CONFIG_PATH`, and `FEATURES` and `SEED` are unchanged.
- [ ] `components/tuning_config.json` has `n_trials` 30, `cv_folds` 5 and the five families in this order: `logreg`, `random_forest`, `hist_gradient_boosting`, `xgboost`, `lightgbm`. The `lightgbm` block has `subsample_freq` 1 in `fixed`.
- [ ] `validate_config` passes on the default file and raises on each of these: a class outside `sklearn.`, `xgboost.` and `lightgbm.`, a class without `predict_proba`, a parameter the class does not have, `low` >= `high`, a parameter in both `fixed` and `search`, an empty `choices`, and a config with no families.
- [ ] For each of the five families, `build_pipeline` fits on a sample of `data/train.csv` and `predict_proba` returns shape `(n, 2)`, with `random_state` equal to `SEED`.

### Components (local)
- [ ] `load_config` returns the five family names in the file order, and raises on an invalid copy.
- [ ] `tune` returns a `result_json` with `family`, `best_params`, `best_cv_auc` and `run_id`. Against a temporary MLflow file store, each family has a parent run with `kind=family` and the artifact `tuning_config.json`, and each trial has a nested run with `kind=trial` and a `trial_state`.
- [ ] `best_cv_auc` of `logreg` is at least 0.87, and `n_complete + n_pruned` equals the number of trials.
- [ ] `grep -c holdout components/tune.py` prints 0: Tune never reads the hold-out.
- [ ] `evaluate` logs the eight `holdout_*` metrics in each parent run and creates the `dummy` run with `holdout_roc_auc` of 0.5.
- [ ] `holdout_tn + holdout_fp + holdout_fn + holdout_tp` equals 3,000, and the winner is the family with the highest `best_cv_auc`.

### Pipeline on the cluster
- [ ] `kubectl get cm churn-code -n kubeflow` holds the four keys `config.py`, `preprocessing.py`, `models.py` and `tuning_config.json`.
- [ ] The compiled pipeline YAML references `churn-base:0.3.0` and does not reference `0.2.0`, and its `ParallelFor` has `parallelism: 2`.
- [ ] The run with 3 trials finishes `Succeeded`, and MLflow shows, for that run (filtering by `kfp_run_id`), five parent runs with three nested runs each, the `holdout_*` metrics in each parent and the `dummy` run. The logs show no `database is locked`.
- [ ] The full run finishes `Succeeded`, and for that run (filtering by `kfp_run_id`) each family has 30 nested runs with `n_complete + n_pruned` equal to 30.
- [ ] In the full run, every family has `holdout_roc_auc` above the `dummy` run, and the winner's `holdout_roc_auc` is at least 0.85 (the CV baseline of LogReg is 0.8789).
- [ ] Every run of the full run has the tag `kfp_run_id`.
- [ ] No model was registered or logged: the registry is empty and no run has a model artifact.
- [ ] `kubectl get pods -A --no-headers | awk '$4!="Running" && $4!="Completed"'` prints nothing, and no temporary pod is left in `kubeflow`.

### Docs and hygiene
- [ ] The `DECISIONS.md` log has dated lines for the `0.3.0` tag and the `xgboost` and `lightgbm` pins, and `git diff main -- DECISIONS.md` shows no change to the frozen contracts table.
- [ ] `docs/runbook.md` has a "Config-driven tuning and evaluation" section with the commands run, the observed results, the times of both runs and the gotchas.
- [ ] The `churn-base` row of README §4 says `0.3.0`, the `Models` and `Tuning` rows of §5 mention the config file, and §7 has the serving risk of XGBoost and LightGBM.
- [ ] `git show HEAD:components/tuning_config.json` has `"n_trials": 30` (the 3-trial edit was not committed), and `git ls-files | grep -c '\.csv$'` returns 0.

## Decisions

- **Yes:** one spec for E2.4, E2.5 and the `ParallelFor` wiring. Evaluate needs the output of Tune, and together they leave a run that shows which family wins.
- **No:** two specs, Tune first and Evaluate after. The first cluster run would show Optuna and MLflow working but nothing to compare against, and Evaluate adds little risk of its own.
- **Yes:** the winner is the family with the best CV ROC-AUC from Tune. Evaluate only reports the hold-out, so no choice is made looking at those 3,000 rows and the gate of SPEC 06 reads a number that took no part in any selection.
- **No:** choosing the winner by hold-out AUC, as the backlog wording suggests. With five candidates it adds a selection bias to the number the gate uses.
- **Yes:** no model is logged in MLflow in this spec. Only Register (SPEC 06) logs and registers the final model, so the 2Gi artifact PVC stays small and no unservable model gets registered.
- **No:** logging the pre-refit model of each family in Evaluate, as the README flow says. It would store models that are never served. The README flow is out of date on this point and the README is updated in step 11.
- **Yes:** Evaluate refits the best config on all of `train.csv` and scores the hold-out with that model. It does not use the CV fold models, which are fitted on 80 % of train and are not kept. The model lives only in the memory of the Evaluate pod.
- **Yes:** Evaluate reopens the parent run of each family with the `run_id` that Tune returned and logs the hold-out metrics there, and the Dummy gets its own run `dummy`. SPEC 06 can then read the champion's hold-out AUC from one run.
- **No:** a single `evaluate` run with the metrics of all the families. The numbers would sit apart from the params and CV metrics of the family they describe.
- **Yes:** every run carries the tag `kfp_run_id` (from `dsl.PIPELINE_JOB_ID_PLACEHOLDER`), so runs of different pipeline runs in the same experiment can be told apart.
- **Yes:** the families, ranges and trials live in `components/tuning_config.json`, read at run time by a `load_config` step that validates it and feeds `ParallelFor`. A change is an edit and a new submit, with no recompile and no change to Python.
- **No:** reading the JSON at compile time. The YAML would go stale after an edit and the cause would be hard to see.
- **Yes:** the file sits next to `config.py` and travels in the ConfigMap `churn-code`, refreshed by `tools/sync_code.sh`. It is the same pattern as SPEC 04.
- **Yes:** `n_trials` and `cv_folds` are in the JSON, `n_trials` global with an optional value per family. The smoke run is one edit of the file (3 trials).
- **No:** `n_trials` as a pipeline parameter. It would be a second place to set the same number.
- **Yes:** `class` accepts any class whose path starts with `sklearn.`, `xgboost.` or `lightgbm.` and has `predict_proba`, and `load_config` checks the class and every parameter name before the run starts. A typo fails in seconds and not after 20 minutes.
- **No:** a closed list of classes in `models.py`. Each new family would need a code change, which is what the JSON avoids.
- **No:** any importable class. A path outside the three libraries is almost always a typo.
- **Yes:** five families by default: the three scikit-learn ones plus XGBoost and LightGBM, with 30 trials each and `parallelism=2`. The user asked to try both boosting libraries.
- **Yes:** rebuild the image as `churn-base:0.3.0` with `xgboost`, `lightgbm` and `libgomp1`, and move every component to it. The README asks for one custom image, and the new tag keeps the change traceable.
- **No:** `packages_to_install` in the Tune step. Every pod would download packages at start.
- **No:** a second image only for tuning. Two images to keep in step for a gain of a few hundred MB.
- **Yes:** keep `churn-base:0.2.0` in minikube. Removing it is out of scope.
- **Yes:** `n_jobs: 1` fixed for the tree families and `parallelism=2`. Each pod uses one core, so two pods in parallel do not fight over the node, and the SQLite risk of the README stays at two writers.
- **Yes:** `subsample_freq: 1` fixed for LightGBM. With its default of 0, LightGBM ignores `subsample`, and a trial would seem to vary a parameter that does nothing.
- **Yes:** the ranges of the data model. They are a starting point, and a range is edited in the JSON.
- **Yes:** `SEED`, the split, `FEATURES`, the sampler (TPE) and the pruner (`MedianPruner`) stay in the code and out of the JSON. The first three are frozen contracts, and the other two are not worth a knob yet.
- **Yes:** `components/models.py` is shared by Tune and Evaluate now and by Register in SPEC 06, so the three build the same pipeline from the same config. It sets `random_state=SEED` when the class accepts it.
- **Yes:** Tune returns one `result_json` string. `dsl.Collected` then hands Evaluate a single list, which is simpler than collecting four outputs.
- **Yes:** recall, F1 and the confusion matrix use a threshold of 0.5, and the matrix is logged as four metrics (`holdout_tn`, `holdout_fp`, `holdout_fn`, `holdout_tp`). They are for reporting, because the gate uses ROC-AUC.
- **Yes:** the tasks set `MLFLOW_HTTP_REQUEST_MAX_RETRIES=10`, as the README promised for the SQLite risk.
- **Yes:** the local checks of steps 6 and 7 use a temporary MLflow file store, and the first contact with the real server is the run of step 9. A bug in the logic then shows up before the cluster is involved.
- **Yes:** the first cluster run uses 3 trials. It checks MLflow from the pods, `ParallelFor` and SQLite in minutes, before the full run.
- **Yes:** a champion of XGBoost or LightGBM cannot be served by the `sklearnserver` image yet. The spec of Pipeline 2 builds a derived image with the same versions, and the risk is recorded here.
- **No:** a `registrable: false` flag per family. It would make the gate and Register respect it, and the winner would stop being the best by CV.
- **No:** leaving XGBoost and LightGBM for later. The user asked for them now.
- **No:** Katib or an Optuna database (RDB). Each study lives in its pod, and the results live in MLflow, as the README says.
- **No:** a change to the frozen contracts. The tag and the pins go in the `DECISIONS.md` log (step 1).

## Risks

| Risk | Mitigation |
|---|---|
| LightGBM or XGBoost fails to import in the image on arm64 (`libgomp.so.1: cannot open shared object file`), because `python:3.11-slim` does not carry `libgomp1`. | The `Dockerfile` installs it (step 1), and step 2 fits a tiny model of each library inside a pod with `0.3.0`. If it still fails, stop and fix the image before touching the components. |
| Adding the two libraries moves `scikit-learn`, `numpy` or `scipy` in the image. `scikit-learn` is pinned to the version of the KServe `sklearnserver`, and a different one breaks serving. | Step 1 checks the diff of `requirements.txt`: `scikit-learn` stays at `1.5.2` and `numpy` at `2.4.6`. If `uv` moves either, stop and pin it before rebuilding. |
| On the laptop, `import lightgbm` fails because macOS has no OpenMP library, and the local checks of steps 1 to 7 cannot run. | `brew install libomp`. It only affects the local checks, not the image, and the runbook records it. |
| The champion is an XGBoost or a LightGBM and the `sklearnserver` image cannot load it (`ModuleNotFoundError`), so Pipeline 2 fails after a good training run. | Not a problem of this spec, which neither registers nor serves. The spec of Pipeline 2 builds a derived `sklearnserver` image with the same versions as `churn-base:0.3.0`. The README §7 records the risk (step 11). |
| SQLite under writes: five families with 30 trials each make about 155 runs, against the 90 the README planned, so `database is locked` or slow calls are more likely. | `parallelism=2` and `MLFLOW_HTTP_REQUEST_MAX_RETRIES=10`. Step 9 looks for the error with 3 trials and step 10 with 30. If it shows up, stop, save the logs, and the fallback is Postgres, in its own spec. |
| The node is tight (about 8 GB, with KFP, KServe and MLflow). Two `tune` pods in parallel, one of them a Random Forest of 400 trees, may be `OOMKilled` or push the node into the thrashing of SPEC 03. | Limit of 1.5Gi per `tune` pod, `n_jobs: 1` and `parallelism=2`. The criteria check that no pod leaves `Running`. If a pod is killed, lower the `n_estimators` range in the JSON before touching `parallelism`. |
| The estimate of 15 to 20 minutes per run is wrong, and the full run takes much longer. | Step 10 measures it and the runbook records it. If it is too slow, lower `n_trials` for the heavy families in the JSON, which needs no code change. |
| `dsl.Collected` over `ParallelFor` does not carry the list of `result_json` strings as described in KFP 2.15. | Step 8 compiles and inspects the YAML before the cluster. If it does not work, stop and bring the options to the user. |
| KFP serializes only the component function. A module-level `import models` works with `python_func` on the laptop and fails in the pod with `ModuleNotFoundError`. | Same rule as SPEC 04: imports go inside the body. Step 9 shows the error in the log of the step if it slips through. |
| The 3-trial run and the full run share the experiment `churn-training`, and a failed step leaves runs of a partial run. | Every run has `kfp_run_id`, and the checks filter by it. Runs open inside `with` blocks, so a failure ends them as `FAILED` and not `RUNNING`. The smoke-run runs stay as history. |

## What is **not** in this spec

- Gate, Register, the alias `champion` and the second run (SPEC 06).
- Logging or registering any model in MLflow.
- Serving a champion of XGBoost or LightGBM (the derived `sklearnserver` image goes in the spec of Pipeline 2).
- Choosing the winner by hold-out AUC.
- `SEED`, the split, `FEATURES`, the sampler or the pruner in the JSON.
- A different preprocessor per family.
- Libraries other than scikit-learn, XGBoost and LightGBM.
- A shared Optuna database (RDB), Katib and early stopping inside XGBoost or LightGBM.
- Postgres instead of SQLite for MLflow.
- Removing `churn-base:0.2.0` from minikube.
- Pipeline 2 (deploy), RBAC, monitoring and the load generator.

Each one of those, if it lands, goes in its own spec.
