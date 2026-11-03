# Decisions

Single source of the frozen contracts. Changing a row needs a new line in the log below **first**, then the code and docs.

## Frozen contracts

| Contract | Value |
|---|---|
| Target | `Exited` (0/1), output is a probability |
| Dropped columns | `id`, `CustomerId`, `Surname` (`id` kept aside only for the submission) |
| `FEATURES` order | `CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary` |
| Split | Stratified 80/20 train/hold-out, `SEED = 42` |
| Main metric | ROC-AUC (plus PR-AUC, recall, F1, confusion matrix) |
| Gate | Hold-out ROC-AUC > Dummy baseline + 0.05 **and** ≥ current champion's hold-out AUC |
| Registry | Model `churn`, alias `@champion` |
| scikit-learn | `1.5.2`, pinned to the KServe 0.21 sklearnserver (`kserve/sklearnserver:v0.21.0`, Python 3.11.16) |

## Log

- **2026-10-30 (SPEC 01):** contracts above written from README §2 and §5. `SEED = 42` chosen because the README asked for a fixed seed without a value.
- **2026-10-30 (SPEC 01, E0.2):** scikit-learn pinned to `1.5.2` and Python to `3.11`, read from the sklearnserver image (see `docs/runbook.md`). The server also ships numpy 2.4.4, which SPEC 02 must verify against a model pickled by `churn-base`.
- **2026-10-30 (SPEC 01, E0.4/Python env):** runtime depends on `mlflow-skinny` instead of `mlflow`. The full package pulls `matplotlib` into the image, which the spec keeps in the `dev` group only. The MLflow server (E1) uses its own image, so it is unaffected. `skops` is added explicitly as a runtime dependency: `mlflow.sklearn.save_model` imports it (MLflow's default, safer-than-pickle format) and only the full package pulled it in. Verified with a `save_model`/`load_model` roundtrip.
- **2026-10-31 (SPEC 03, E1.1):** MLflow server pinned to `3.17.2`, the same version as the `mlflow-skinny` client in `churn-base`, using the official image `ghcr.io/mlflow/mlflow:v3.17.2` (no custom build). Deployed in namespace `mlflow` with SQLite and artifacts on a PVC.
- **2026-10-31 (SPEC 03, E1.2):** the MLflow server runs with `--workers 1` and `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`. With the defaults, MLflow 3.17.2 starts several uvicorn workers plus about 6 `huey` job consumers (about 160 MB each), over 1.5 GB in total. That got the container `OOMKilled` at 1Gi and left the 8 GB minikube node thrashing (load above 60, apiserver timeouts). Neither the job executor nor the extra workers are needed here.
- **2026-10-31 (SPEC 03, E1.3):** the MLflow server runs with `--allowed-hosts mlflow.mlflow.svc.cluster.local:5000,mlflow.mlflow.svc.cluster.local,localhost:*,127.0.0.1:*`. MLflow 3.17.2 validates the `Host` header and answered 403 "Invalid Host header" to calls for `mlflow.mlflow.svc.cluster.local:5000` from the `mlflow-check` pod. `localhost:*` and `127.0.0.1:*` keep the `kubectl port-forward` UI working. No wildcard `*`.
- **2026-11-02 (SPEC 04, E2.2):** `kfp` pinned to `2.15.0`, the same version as the KFP backend, and added to the runtime dependencies of `churn-base`. KFP lightweight components run `kfp` inside the step pod, and `churn-base:0.1.0` does not carry it. `kfp-kubernetes==2.15.0` goes in the `dev` group because it is only needed to compile the pipeline.
- **2026-11-02 (SPEC 04, E2.8):** image tag `churn-base:0.2.0` (`0.1.0` plus `kfp`). `0.1.0` stays in the cluster untouched, so SPEC 02 and SPEC 03 keep working.
- **2026-11-02 (SPEC 05, E2.4):** image tag `churn-base:0.3.0` (`0.2.0` plus `xgboost`, `lightgbm` and the system library `libgomp1`, which LightGBM needs and `python:3.11-slim` does not carry). `0.2.0` stays in the cluster untouched. Every pipeline component moves to `0.3.0`.
- **2026-11-02 (SPEC 05, E2.4):** `xgboost` pinned to `3.2.0` and `lightgbm` to `4.7.0`, added to the runtime dependencies of `churn-base`. `scikit-learn` stays at `1.5.2` and `numpy` at `2.4.6`. The KServe `sklearnserver:v0.21.0` image has neither library, so a champion of those families cannot be served by it: the Pipeline 2 spec must build a derived image with these same versions.
