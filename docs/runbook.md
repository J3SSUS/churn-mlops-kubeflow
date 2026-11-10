# Runbook — Bank Customer Churn MLOps

Commands actually run, in order. Sections are added as each spec is implemented.

## Cluster (SPEC 01, E0.1)

Prerequisite: Docker Desktop running (`docker info` answers). Docker Desktop memory was ~8 GB.

```bash
minikube start
kubectl get pods -n kubeflow
kubectl get pods -n kserve
kubectl get pods -A --no-headers | awk '$4!="Running" && $4!="Completed"'   # must print nothing
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80                 # UI at http://localhost:8080
```

Observed on 2026-10-30:

- minikube v1.39.0, Kubernetes v1.32.0 (containerd), driver `docker`, arm64.
- After a cold `minikube start`, pods restart for about a minute (`metadata-grpc` reached 20 restarts in total, which is history from earlier sessions). All 19 pods in `kubeflow` and all 3 in `kserve` were `Ready` after ~1 minute.
- `GET http://localhost:8080/` returned 200 and `/apis/v2beta1/healthz` returned `apiServerReady: true`.

Gotchas:

- With Docker Desktop closed, `minikube status` fails with "failed to connect to the docker API" and `kubectl` reports `current-context is not set`. Open Docker Desktop first.
- `kubectl` is v1.37 against a v1.32 cluster. It only warns; nothing broke.

## KServe sklearnserver versions (SPEC 01, E0.2)

The runtime image comes from the `kserve-sklearnserver` ClusterServingRuntime. Versions were read by running the image as a throwaway pod (no host Docker needed):

```bash
kubectl get clusterservingruntime kserve-sklearnserver -o jsonpath='{.spec.containers[*].image}'
kubectl run sklearn-versions -n default --image=kserve/sklearnserver:v0.21.0 --restart=Never \
  --command -- /prod_venv/bin/python -c 'import sys, sklearn, joblib, numpy, scipy; print(sys.version.split()[0], sklearn.__version__, joblib.__version__, numpy.__version__, scipy.__version__)'
kubectl logs sklearn-versions -n default
kubectl delete pod sklearn-versions -n default
```

| Item | Value |
|---|---|
| Image | `kserve/sklearnserver:v0.21.0` (digest `sha256:8967d3fa3a8132e5e4d0dd7fc743214c451f4d9481edb0ab306e657149fd5a06`) |
| Python (server venv) | 3.11.16 |
| scikit-learn | **1.5.2** |
| joblib / numpy / scipy | 1.5.3 / 2.4.4 / 1.17.1 |
| Architecture | `aarch64` (the pod ran natively on the arm64 minikube node, so there is an arm64 variant) |

Gotchas:

- The server runs in `/prod_venv`, not the system Python. `pip show scikit-learn` with the default `pip` reports "not found"; use `/prod_venv/bin/python`.
- `numpy` is 2.4.x in the server. A model pickled under another numpy major may not load, so `churn-base` should pin numpy through the scikit-learn 1.5.2 resolution and SPEC 02 must check the load.
- `docker manifest inspect` was not needed: running the image on the arm64 node already proved the platform.

## Python environment (SPEC 01)

`pyproject.toml` + `uv.lock` are the single source of versions. Runtime dependencies: `scikit-learn==1.5.2` (the sklearnserver version), `mlflow-skinny`, `skops`, `optuna`, `pandas`, `joblib`. `jupyter` and `matplotlib` live in the `dev` group.

```bash
uv lock
uv sync
uv run python -c "import sklearn, mlflow, optuna, pandas, joblib; print(sklearn.__version__)"   # 1.5.2
uv export --no-dev --no-hashes --format requirements-txt | grep -ciE '^(jupyter|matplotlib)[=<>~ ]'   # 0
```

Gotchas:

- The full `mlflow` package pulls `matplotlib` into the runtime set, so the project uses `mlflow-skinny`.
- `mlflow.sklearn.save_model` imports `skops` (MLflow's default format), and only the full package installed it. It is a direct dependency now. Checked with a `save_model`/`load_model` roundtrip.
- Switching `mlflow` to `mlflow-skinny` left `import mlflow` broken (shared files were removed on uninstall). Fix: `uv sync --reinstall-package mlflow-skinny`.
- MLflow detects the uv project and builds a model's requirements from `uv.lock`. Pipeline pods have no `uv.lock`, so check the registered model's `requirements.txt` pins `scikit-learn==1.5.2` (SPEC 02).

## `churn-base` image (SPEC 01, E0.4)

`docker/churn-base/requirements.txt` is generated, never edited by hand. The `Dockerfile` uses `python:3.11-slim`, installs the requirements and has no `CMD`.

```bash
uv export --no-dev --no-hashes --no-emit-package nvidia-nccl-cu12 --format requirements-txt > docker/churn-base/requirements.txt
minikube image build -t churn-base:0.1.0 docker/churn-base      # builds inside minikube (containerd)
minikube image ls | grep churn-base                             # docker.io/library/churn-base:0.1.0
kubectl run churn-base-test -n kubeflow --image=churn-base:0.1.0 --image-pull-policy=IfNotPresent \
  --restart=Never --command -- python -c 'import sklearn, mlflow, optuna, pandas, joblib; print(sklearn.__version__)'
kubectl logs churn-base-test -n kubeflow
kubectl delete pod churn-base-test -n kubeflow
```

Observed: the build takes about a minute; the pod printed scikit-learn 1.5.2 (Python 3.11.18, numpy 2.4.6, mlflow 3.17.2).

Gotchas:

- `minikube image build` builds inside minikube, so the image does not show in the host's `docker images`. Check with `minikube image ls`.
- The tag goes up (`0.1.1`, ...) whenever `pyproject.toml` or the `Dockerfile` changes. Pods use `IfNotPresent`, so reusing a tag would keep the stale image.
- The image has Python 3.11.18; the sklearnserver has 3.11.16. Same minor version, so pickles stay compatible.

## Data PVC (SPEC 01, E0.5)

PVC `churn-data` (2Gi, `ReadWriteOnce`, storageClass `standard`) in `kubeflow`, mounted at `/mnt/data`. A helper pod copies the CSVs from `data/` to `/mnt/data/raw/` and is deleted afterwards.

```bash
kubectl apply -f k8s/pvc-churn-data.yaml -f k8s/data-loader-pod.yaml
kubectl wait --for=condition=Ready pod/data-loader -n kubeflow --timeout=120s
kubectl exec -n kubeflow data-loader -- mkdir -p /mnt/data/raw
for f in train test sample_submission; do kubectl cp data/$f.csv kubeflow/data-loader:/mnt/data/raw/$f.csv; done
kubectl exec -n kubeflow data-loader -- ls -l /mnt/data/raw
kubectl exec -n kubeflow data-loader -- sh -c 'cd /mnt/data/raw && wc -l *.csv'    # 10001 sample_submission, 10001 test, 15001 train
kubectl delete pod data-loader -n kubeflow                                          # the PVC stays
kubectl get pvc churn-data -n kubeflow                                              # Bound, 2Gi
```

Observed: the PVC was `Bound` in seconds; the 3 files match the local `md5` checksums.

Gotchas:

- The PVC uses minikube's hostpath provisioner (`Delete` reclaim policy), so `minikube delete` or deleting the PVC loses the data. Reload it by re-running the commands above; the CSVs stay in `data/`.
- Pods can only mount a PVC from their own namespace, so everything that needs `churn-data` runs in `kubeflow`.

## Serving spike (SPEC 02, E0.7)

Goal: check how a sklearn `Pipeline` trained in `churn-base:0.1.0` answers raw rows on a temporary KServe InferenceService. Everything below was deleted at the end.

```bash
# 1. Train in a churn-base pod (the code travels in a ConfigMap, the image is not rebuilt)
kubectl create configmap spike-code -n kubeflow --from-file=components/config.py \
  --from-file=tools/spike_pipeline.py --from-file=tools/spike_train.py --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/spike-train-pod.yaml
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/spike-train -n kubeflow --timeout=180s
kubectl logs spike-train -n kubeflow | grep '^REFERENCE ' | sed 's/^REFERENCE //' > spike-reference.json   # git-ignored
kubectl exec -n kubeflow data-loader -- ls -la /mnt/data/models/spike/v1                                    # only model.joblib

# 2. Deploy and wait
kubectl apply -f k8s/spike-inferenceservice.yaml
kubectl wait --for=condition=Ready isvc/churn-spike -n kubeflow --timeout=300s

# 3. Query (use another local port if 8080 is taken)
kubectl port-forward -n kubeflow svc/churn-spike-predictor 8080:80
uv run python tools/spike_query.py            # or: uv run python tools/spike_query.py 18080
```

Facts recorded before the InferenceService was created:

- The training pod ran `churn-base:0.1.0` (Python 3.11.18, scikit-learn 1.5.2, joblib 1.6.0, numpy 2.4.6).
- `models/spike/v1/` held exactly one file, `model.joblib` (2,990 bytes).
- `spike-reference.json` held 3 instances of 10 values each, with strings at positions 1 and 2 (Geography, Gender), plus 3 rows of `expected`.

### Findings

| Question | Answer (yes/no) | Evidence |
|---|---|---|
| `PREDICT_PROBA=true` returns probabilities | yes | HTTP 200, `{"predictions":[[0.7844881096085148,0.21551189039148516],[0.924457286540825,0.07554271345917493],[0.5147346067488787,0.4852653932511213]]}` |
| A raw row with strings and numbers keeps its types and is scored | yes | The 3 rows with `"Germany"`/`"France"` and `"Male"`/`"Female"` at positions 1 and 2 returned 200 with no error, and the probabilities equal the local ones |
| The model loads and matches the local `predict_proba` despite the numpy/Python patch differences | yes | Maximum absolute difference `0.000e+00` (threshold `1e-6`). Pod: Python 3.11.18, numpy 2.4.6. Server: Python 3.11.16, numpy 2.4.4 |
| It runs in Standard mode with the explicit `kserve-sklearnserver` runtime | yes | `kubectl get isvc churn-spike -n kubeflow` showed `READY=True`; runtime `kserve-sklearnserver`, deployment mode `Standard`, predictor pod `1/1 Running` with image `kserve/sklearnserver:v0.21.0`; the log ends with `Application startup complete` |

Real answer shape (`tools/spike_query.py` output):

```text
status: 200
predictions shape: (3, 2)
max abs diff vs expected: 0.000e+00
```

- `predictions` is a list with one `[p0, p1]` pair per instance. **The churn probability is the second column, `p1`.** E3.4 and E4.4 must read `predictions[i][1]`.
- The request is `{"instances": [[649.0, "Germany", "Male", 37.0, 3.0, 106534.4, 1.0, 1.0, 1.0, 131173.9], ...]}`: numbers as floats, strings as strings, in `FEATURES` order.

Gotchas:

- If another process holds local port 8080 (for example the KFP UI port-forward from the cluster section), a `POST` to `localhost:8080` reaches that process and returns an HTML `404 Cannot POST`. Forward the predictor to a free port and pass it to `spike_query.py`.
- `sklearnserver` takes exactly one `.joblib`, `.pkl` or `.pickle` file per model folder, so nothing else may go in `models/spike/v1/`.
- Pods are immutable: after editing a spike script, recreate the ConfigMap and delete `spike-train` before running it again.

### Cleanup

```bash
kubectl delete -f k8s/spike-inferenceservice.yaml
kubectl delete pod spike-train -n kubeflow
kubectl delete configmap spike-code -n kubeflow
kubectl exec -n kubeflow data-loader -- rm -rf /mnt/data/models
kubectl exec -n kubeflow data-loader -- ls /mnt/data                    # only raw
kubectl delete pod data-loader -n kubeflow
```

## MLflow (SPEC 03, E1)

Goal: run an MLflow tracking server with model registry in the `mlflow` namespace, and check that a `churn-base` pod in `kubeflow` can log a run with an artifact and register, alias and load a test model. The server, its PVC and its Service stay installed (E2 builds on them). Everything else below was deleted at the end.

```bash
# 1. Deploy and wait (server 3.17.2, official image, SQLite + artifacts on the PVC)
kubectl apply -f k8s/mlflow.yaml
kubectl wait --for=condition=Available deploy/mlflow -n mlflow --timeout=300s

# 2. UI on a free local port (8080 is held by the KFP UI port-forward, 5000 by AirPlay on macOS)
kubectl port-forward -n mlflow svc/mlflow 5001:5000
curl http://localhost:5001/health                       # OK

# 3. Check pod (the code travels in a ConfigMap, the image is not rebuilt)
kubectl create configmap mlflow-check-code -n kubeflow --from-file=tools/mlflow_check.py
kubectl apply -f k8s/mlflow-check-pod.yaml
kubectl wait --for=condition=Ready pod/mlflow-check -n kubeflow --timeout=120s
kubectl exec -n kubeflow mlflow-check -- python /code/mlflow_check.py connectivity
kubectl exec -n mlflow deploy/mlflow -- find /mlflow/artifacts -type f

# 4. Registry (recreate the ConfigMap and the pod first if the script changed)
kubectl exec -n kubeflow mlflow-check -- python /code/mlflow_check.py registry

# 5. Cleanup of the test objects (the server stays)
kubectl exec -n kubeflow mlflow-check -- python /code/mlflow_check.py cleanup
kubectl delete pod mlflow-check -n kubeflow
kubectl delete configmap mlflow-check-code -n kubeflow
```

Observed:

- The official image `ghcr.io/mlflow/mlflow:v3.17.2` runs on arm64 (no `exec format error`).
- The PVC `mlflow-data` was `Bound` with 2Gi, the Deployment `1/1`, and `/health` answered `OK`.
- `connectivity` exited 0, printed a run id and the artifact URI `mlflow-artifacts:/1/<run id>/artifacts`, and read `hello.txt` back with the same content. The run held the param `check=connectivity` and the metric `answer=42.0` (read through the API). `hello.txt` was in `/mlflow/artifacts/1/<run id>/artifacts/` on the server's PVC, so the artifact travelled through the server and not through a client-side write.
- `registry` printed `model version: 1` and `max absolute difference: 0.0`. The registry held only `registry-test`, with the alias `check` on version 1 (read through the API). The `mlflow-skinny` client in `churn-base` needed nothing extra.
- UI check: the runs of `connectivity-test` were seen in the UI at `http://localhost:5001` before the cleanup, and they had run correctly. No screenshot was kept, because the cleanup deleted them afterwards. The model `registry-test` and its alias were verified through the API only.
- After `cleanup` the registry was empty, `connectivity-test` was in the `deleted` stage, no `mlflow-check*` object was left in `kubeflow`, and the server pod, the PVC and the Service were still there.

Gotchas:

- **Memory:** with its defaults MLflow 3.17.2 starts several uvicorn workers and about 6 `huey` job consumers (about 160 MB each), over 1.5 GB in total. At a 1Gi limit the pod was `OOMKilled` (`CrashLoopBackOff`, exit 137), and at 2Gi it pushed the 8 GB node into thrashing (load above 60, `kubectl` and `minikube status` timing out) until `minikube stop && minikube start`. The manifest runs `--workers 1` and `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`, and the pod uses about 550 MB with a 1Gi limit. If `kubectl` hangs, `docker exec minikube sh -c "top -b -n 1 -o %CPU | head"` shows what is eating the node. After a minikube restart a few controllers show `Error` or `CrashLoopBackOff` for a minute and recover alone.
- **Host header:** MLflow 3.x validates it and answers `403 Invalid Host header - possible DNS rebinding attack detected` to `mlflow.mlflow.svc.cluster.local:5000`. The server runs with `--allowed-hosts mlflow.mlflow.svc.cluster.local:5000,mlflow.mlflow.svc.cluster.local,localhost:*,127.0.0.1:*` (see `DECISIONS.md`). Any other in-cluster name needs adding there.
- **Soft deletes:** the API cleanup leaves the experiment `connectivity-test` and its runs as soft-deleted rows (no `mlflow gc`), so that name cannot be reused. To re-run the checks, change `EXPERIMENT` in `tools/mlflow_check.py`.
- **Client noise:** the `churn-base` image has no `git`, so the client prints a `GIT_PYTHON_REFRESH` warning. It is harmless.
- **Port-forward:** it dies when the server pod is replaced (any `kubectl apply` that changes the Deployment). Start it again.
- **Local `curl` with query strings:** quote the URL in zsh (`curl 'http://localhost:5001/api/...?name=x'`), or the `?` is taken as a glob.
- The PVC uses minikube's hostpath provisioner, so `minikube delete` loses every run and model. E2 can recreate them.
- Pods are immutable: after editing `mlflow_check.py`, recreate the ConfigMap and delete `mlflow-check` before running it again.

## Training pipeline skeleton (SPEC 04, E2.1-E2.3 and E2.8 first part)

Pipeline `churn-training` with two steps, Ingest and Preprocess. Ingest validates `/mnt/data/raw/train.csv`; Preprocess drops the identifiers, makes the stratified 80/20 split with `SEED = 42` and writes `train.csv` and `holdout.csv` to `/mnt/data/processed` on the PVC. No MLflow call yet. The shared code (`config.py`, `preprocessing.py`) travels in the ConfigMap `churn-code`, mounted at `/code` with `PYTHONPATH=/code`, so a code change does not need an image rebuild.

```bash
# 1. Dependencies and image (kfp is now a runtime dependency; kfp-kubernetes is dev only)
uv lock
uv export --no-dev --no-hashes --no-emit-package nvidia-nccl-cu12 --format requirements-txt > docker/churn-base/requirements.txt
minikube image build -t churn-base:0.2.0 docker/churn-base
minikube image ls | grep churn-base                      # 0.1.0 and 0.2.0
kubectl run churn-base-test -n kubeflow --image=churn-base:0.2.0 --image-pull-policy=IfNotPresent \
  --restart=Never --command -- python -c 'import kfp, sklearn, mlflow; print("kfp", kfp.__version__, "| scikit-learn", sklearn.__version__, "| mlflow", mlflow.__version__)'
kubectl wait -n kubeflow --for=jsonpath='{.status.phase}'=Succeeded pod/churn-base-test --timeout=60s
kubectl logs churn-base-test -n kubeflow
kubectl delete pod churn-base-test -n kubeflow

# 2. Compile only (no cluster needed), then submit
uv run python tools/submit_pipeline.py --compile-only
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888      # in another terminal
uv run python tools/submit_pipeline.py --wait                   # syncs the ConfigMap, compiles, submits

# 3. Check the split on the PVC (temporary pod)
kubectl apply -f k8s/data-loader-pod.yaml
kubectl wait -n kubeflow --for=condition=Ready pod/data-loader --timeout=90s
kubectl exec -n kubeflow data-loader -- sh -c 'ls -l /mnt/data/processed; wc -l /mnt/data/processed/*.csv'
kubectl delete pod data-loader -n kubeflow
```

Observed:

- The pod with `churn-base:0.2.0` printed `kfp 2.15.0 | scikit-learn 1.5.2 | mlflow 3.17.2`. The build took about a minute.
- The change in `docker/churn-base/requirements.txt` only added `kfp` and its dependencies and moved `protobuf` from 7.36.2 to 6.33.6. `scikit-learn` stayed at `1.5.2`.
- The compiled YAML references the image `churn-base:0.2.0`, the PVC `churn-data` at `/mnt/data` and the ConfigMap `churn-code` at `/code`, with `imagePullPolicy: IfNotPresent`.
- The run finished `SUCCEEDED` with both steps green, about 40 seconds from submit to the end.
- Ingest logged `ingest ok: 15000 rows, 14 columns, 0 nulls`. Preprocess logged `train 12000 rows (20.3500% positive), holdout 3000 rows (20.3667% positive)`.
- The PVC held `processed/train.csv` with 12,001 lines and `processed/holdout.csv` with 3,001 (header included). The ConfigMap `churn-code` had the keys `config.py` and `preprocessing.py`.
- MLflow stayed untouched: only the `Default` experiment was active and the registry was empty.
- Locally, Ingest raised on each bad copy of `train.csv` (a missing column, a text `Age`, a numeric `Geography`, one null and `Exited = 2`), and two Preprocess runs gave byte-identical files.

Gotchas:

- **Import name:** the package `kfp-kubernetes` is imported as `kfp.kubernetes` (`from kfp import kubernetes`), not `kfp_kubernetes`. Pin it to the same version as `kfp` (`2.15.0`).
- **Imports inside the function:** KFP serializes only the component function, so `import config` goes inside the body. At module level it works on the laptop and fails in the pod with `ModuleNotFoundError`.
- **Caching:** the KFP cache key does not include the ConfigMap content, so a fixed bug in `config.py` could seem not to run. Every task sets `set_caching_options(False)`.
- **ConfigMap freshness:** `tools/submit_pipeline.py` runs `tools/sync_code.sh` before submitting. If you submit by another route, run `tools/sync_code.sh` first or the pods keep the old code.
- **Two port-forwards:** the API (`svc/ml-pipeline`, port 8888) is what `submit_pipeline.py` needs. The UI (`svc/ml-pipeline-ui 8080:80`) is only for the browser. They are different services.
- **Run it from the repo root:** `pipelines/training_pipeline.py` imports `components.ingest` and `components.preprocess`, and `submit_pipeline.py` adds the repo root to `sys.path` itself.
- **Local test of a component:** call `component.python_func(...)` with `config.RAW_DIR_PVC` or `config.PROCESSED_DIR_PVC` reassigned to a temporary folder. The compiled YAML is git-ignored (`pipelines/*.yaml`) and rebuilt on every submit.
- The pipeline pods mount `churn-data` (`ReadWriteOnce`), so a helper pod can only mount it after the run finishes.

## Config-driven tuning and evaluation (SPEC 05, E2.4-E2.5 and E2.8 second part)

Pipeline `churn-training` now runs Ingest → Preprocess → Load config → `ParallelFor` (Tune per family, `parallelism=2`) → Evaluate. The families, their ranges and the number of trials live in `components/tuning_config.json` (five families by default, 30 trials each). The pods read it from the ConfigMap `churn-code`, which now has four keys: `config.py`, `preprocessing.py`, `models.py` and `tuning_config.json`. Tune logs params and CV metrics in a parent run per family and a nested run per trial. Evaluate reopens each parent run, logs the `holdout_*` metrics and the tag `winner`, and creates the run `dummy`. The winner is the family with the best CV AUC, and no model is logged: only Register (SPEC 06) will do that.

```bash
# 1. Image 0.3.0: xgboost, lightgbm and libgomp1. requirements.txt is exported without nvidia-nccl-cu12 and installed with --no-deps
uv lock
uv export --no-dev --no-hashes --no-emit-package nvidia-nccl-cu12 --format requirements-txt > docker/churn-base/requirements.txt
minikube image build -t churn-base:0.3.0 docker/churn-base
minikube image ls | grep churn-base                      # 0.1.0, 0.2.0 and 0.3.0
kubectl run churn-base-test -n kubeflow --image=churn-base:0.3.0 --image-pull-policy=IfNotPresent \
  --restart=Never --command -- python -c 'import kfp, sklearn, mlflow, xgboost, lightgbm; print(kfp.__version__, sklearn.__version__, mlflow.__version__, xgboost.__version__, lightgbm.__version__)'   # the real check also fits a tiny XGBClassifier and LGBMClassifier and runs pip check
kubectl wait -n kubeflow --for=jsonpath='{.status.phase}'=Succeeded pod/churn-base-test --timeout=90s
kubectl logs churn-base-test -n kubeflow
kubectl delete pod churn-base-test -n kubeflow

# 2. Compile only (no cluster needed), then submit
uv run python tools/submit_pipeline.py --compile-only
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888      # KFP API, in another terminal
kubectl port-forward -n mlflow svc/mlflow 5001:5000             # MLflow UI at http://localhost:5001
uv run python tools/submit_pipeline.py                          # syncs the ConfigMap, compiles, submits
kubectl get workflow <name> -n kubeflow -o jsonpath='{.status.phase}'   # follow the run without the KFP API
```

To find the runs of one pipeline run in MLflow, filter the experiment `churn-training` with `tags.kind = "family" and tags.kfp_run_id = "<run id>"` and sort by `best_cv_roc_auc`. `tags.winner = "true"` finds the winner.

Local checks of Tune and Evaluate (no cluster): call `component.python_func(...)` with `config.MLFLOW_TRACKING_URI` pointed at a temporary directory and `config.TUNING_CONFIG_PATH` at a copy of the JSON with fewer trials. MLflow 3.17.2 blocks the file store unless `MLFLOW_ALLOW_FILE_STORE=true` is set for that run.

Observed:

- The `pip install` step of the build took 87 s with `nvidia-nccl-cu12` and 59 s without it. A pod with `0.3.0` printed `kfp 2.15.0 | scikit-learn 1.5.2 | mlflow 3.17.2 | xgboost 3.2.0 | lightgbm 4.7.0`, and a tiny `XGBClassifier` and `LGBMClassifier` fitted. `pip check` reports exactly one line, `xgboost 3.2.0 requires nvidia-nccl-cu12, which is not installed`, which is the intended omission.
- A run with `n_trials: 3` finished `Succeeded` in 4 minutes (04:08:53 to 04:12:55 UTC): 5 parent runs, 15 nested runs, the `dummy` run, no `database is locked` and no model logged.

  | Family | CV AUC | Hold-out AUC |
  |---|---|---|
  | `hist_gradient_boosting` (winner) | 0.9326 | 0.9349 |
  | `xgboost` | 0.9317 | 0.9318 |
  | `lightgbm` | 0.9289 | 0.9309 |
  | `random_forest` | 0.9266 | 0.9305 |
  | `logreg` | 0.8771 | 0.8869 |
  | `dummy` | — | 0.5000 |

- A boosting model scores about 0.05 AUC above LogReg (0.878 in the README baseline). An independent `HistGradientBoosting` with default parameters gave 0.9315 in 5-fold CV on all 15,000 rows, so the gap is not a leak.
- The run with 30 trials was **not made**: the user will do it at the end of the project, as one longer run. Its time is unknown. The 15 to 20 minutes written in the spec was an estimate and has not been checked.

Gotchas:

- **KFP pods in a new run fail to reach the metadata store.** In about 17 submissions most failed within the first two minutes, in the KFP drivers and launchers, with `GetContextByTypeAndName ... mysql_real_connect failed`, `failed to pre-publish Pod info to ML Metadata ... dial tcp <metadata-grpc-service>:8080: i/o timeout`, or, in the KFP API, `lookup mysql on 10.96.0.10:53: no such host`. The failing pods were KFP's own, before our code ran. MySQL was healthy (3 of 151 connections, no connection errors), CPU was 97 % idle and 2.7 GB of memory was free. A probe pod saw TCP timeouts to the Service IPs and DNS lookups of 7 to 20 s during its first 30 s of life and while other pods were starting. **The cause was not found.** Restarting `metadata-grpc`, `ml-pipeline` and MySQL did not fix it. After the `minikube start` of SPEC 04 and after restarting Docker Desktop the first runs went through and it degraded later, so a restart helps for a while at most. Raising the node CPU quota did not fix it either.
- **What worked was resubmitting.** A run finished on the fourth attempt of a loop that resubmits only when the failed pods show those errors. A Python traceback in a pod is a real failure of our code and is not retried. The loop was a throw-away script, not part of the repo.
- **While the flake lasts, the KFP API also fails.** `tools/submit_pipeline.py --wait` can die with a 500 while the run keeps going, and the KFP UI can show `RUNNING` for a run that Argo already marked as failed. Follow the workflow phase with `kubectl` instead.
- **The node has a CPU quota of 4 cores, but the pods see 12.** `nproc` says 12 and `cpu.max` of the node container says `400000 100000`. `HistGradientBoostingClassifier` has no `n_jobs`, opened 12 OpenMP threads and one trial took 4 min 51 s. Every task now sets `OMP_NUM_THREADS=1`, which is the `n_jobs: 1` rule for the libraries without that parameter. In the first 30 minutes after a Docker restart the node was throttled in 3,858 of 18,223 periods (21 %), 867 s in total (`minikube ssh "cat /sys/fs/cgroup/cpu.stat"`). `docker update --cpus=8 minikube` raises the quota without a restart and the throttling stopped, but the failures went on, so the quota was not the cause. It was not checked whether the new quota survives a `minikube stop` and `start`.
- **`mlflow.set_experiment` has a race in MLflow 3.17.2.** When two processes create the same experiment at the same moment, the loser returns without setting the active experiment and logs to `Default`. Evaluate then fails with `Cannot start run with ID ... because active experiment ID does not match environment run ID`. It happens in the first `ParallelFor` on a fresh server. `models.use_experiment()` sets the experiment by id afterwards: with two processes, the plain call failed in 10 of 10 races and the helper in 0 of 10.
- **A parent run without `holdout_*` metrics means Evaluate did not run** for that pipeline run. Tune never reads the hold-out (`grep -c holdout components/tune.py` prints 0).
- **LightGBM and XGBoost take `**kwargs`**, so `validate_config` cannot see every parameter. `verbose` of LightGBM is in `KNOWN_EXTRA_PARAMS` in `components/models.py`, and `verbose` for XGBoost is rejected on purpose. With its default `subsample_freq` of 0, LightGBM ignores `subsample`, so the config fixes it to 1.
- **`nvidia-nccl-cu12` is left out on purpose.** XGBoost 3.2.0 declares it for Linux, and its `aarch64` wheel is 352 MB of GPU code. Removing the line from `requirements.txt` was not enough, because `pip` resolved it as a dependency of `xgboost`. The `Dockerfile` installs with `pip install --no-deps`.
- **A failed attempt leaves its Argo workflow and its pods in `Error`.** After about 17 submissions there were 16 failed workflows. The hygiene check of the spec (no pod outside `Running` or `Completed`) needs them deleted.
- **The Docker disk was at 89 %** (50 of 59 GB): `docker system df` showed 19.5 GB of images (14.2 GB reclaimable), 26.9 GB of volumes (mostly the minikube node: 23.4 GB in `containerd`) and 7 GB of build cache. `docker image prune -a` only removes images no container uses. Do not run `docker volume prune`, because it can take data from other projects.
- **The port-forwards die** with `minikube stop` and when the pod behind the Service is replaced, because each one is tied to one pod. The API one (8888) is the one `submit_pipeline.py` needs.
- **BusyBox `nslookup` exits with an error** when a search domain answers `NXDOMAIN`, even if the right name resolves. Use the FQDN with a trailing dot when testing DNS with it.

## Gate and Register (SPEC 06, E2.6-E2.7, E2.8 last part and E2.9)

Pipeline `churn-training` now ends with Evaluate → Gate → `dsl.If(passed)` Register. Gate reads the Dummy AUC from the run `dummy` of the same pipeline run (`tags.kind = 'dummy' and tags.kfp_run_id = '<run id>'`) and the champion's AUC from the tag `holdout_roc_auc` of the version behind `churn@champion`, and applies the gate of the frozen contracts (`GATE_MARGIN = 0.05` in `config.py`). A rejection is a normal result: the step succeeds with `passed=false`, Register is skipped and the run ends `Succeeded`. Register refits the winner on train + hold-out, logs it in a run `register`, registers it as `churn`, sets the four tags on the version and only then moves the alias `champion`, so a failure in between can never leave a champion without the tag the next gate reads.

```bash
# 1. Smoke runs use 3 trials: edit "n_trials" in components/tuning_config.json to 3, and put 30 back before committing
uv run python tools/submit_pipeline.py --compile-only           # no cluster needed
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888      # KFP API, in another terminal
kubectl port-forward -n mlflow svc/mlflow 5001:5000             # MLflow UI at http://localhost:5001
uv run python tools/submit_pipeline.py --wait                   # syncs the ConfigMap, compiles, submits, waits up to 15 min
kubectl get workflow <name> -n kubeflow -o jsonpath='{.status.phase}'   # follow the run without the KFP API

# 2. Second run: the same command. The gate now compares against the champion
# 3. Hygiene: prints nothing when every pod is Running or Completed
kubectl get pods -A --no-headers | awk '$4!="Running" && $4!="Completed"'
```

In MLflow, `tags.winner = "true"` finds the winner and its tag `gate` says `passed` or `rejected`. The champion is the version behind the alias `champion`, and its tags (`holdout_roc_auc`, `winner_run_id`, `family`, `kfp_run_id`) say where it came from. In MLflow 3 the model is a **logged model** named `model`, linked to the run `register` by `source_run_id`: `list_artifacts` of that run is empty, and `MlflowClient.search_logged_models(experiment_ids=[...], filter_string="source_run_id = '<run id>'")` finds it.

Local checks of Gate and Register (no cluster): call `component.python_func(...)` with `config.MLFLOW_TRACKING_URI` pointed at `sqlite:///<tmp>/mlflow.db` (the registry with aliases needs a database-backed store, and the experiment must be created with an `artifact_location` in the temporary folder) and `config.TUNING_CONFIG_PATH` at the repo JSON. The split comes from calling `preprocess.python_func` with `config.PROCESSED_DIR_PVC` in the temporary folder. The rejection path uses a tiny `LogisticRegression` registered as `churn@champion` with a `holdout_roc_auc` tag higher than the winner's.

Observed:

- **First champion.** The first run that reached Register (`b58a822a-4bfa-4088-bf74-8606bdfeb1cb`) logged `gate PASSED for hist_gradient_boosting: winner 0.9349 > dummy 0.5000 + 0.05; no champion yet` and created `churn` v1 with the alias `champion`. Ingest to Evaluate took about 4.5 minutes, and Gate plus Register took 2 min 17 s. The model weighs 216 KB on the MLflow volume. It had to be rescued with a run retry (see the gotchas).
- **Second run.** A run after a `minikube stop` and `start` (`de7cc238-81cb-4654-9fa3-4318d54d5c46`) finished `Succeeded` in 5 min 8 s at the first attempt: `gate PASSED ...: winner 0.9349 > dummy 0.5000 + 0.05 and >= champion v1 0.9349`, `churn` v2, alias `champion` → v2, and v1 kept with its tags. The hold-out AUC of v1 and v2 is equal to the last decimal (`0.9348699268811841`), so the tie does not depend on float noise: the same seed and params give the same result in two different pods. The other four families repeat their numbers of the SPEC 05 table.
- **Register works for the five families.** Locally, each family registered and reloaded with `predict_proba` of shape `(n, 2)`. With the MLflow default format (`skops`) XGBoost and LightGBM failed with `untrusted types`, so Register logs with `serialization_format="cloudpickle"` (line of 2026-11-04 in `DECISIONS.md`). That the sklearnserver image can **load** such a champion is still open for the Pipeline 2 spec.
- **The node CPU quota survived a restart.** After `minikube stop` and `start`, `cpu.max` still said `800000 100000` (8 cores), so the `docker update --cpus=8 minikube` of SPEC 05 is not needed again. The node disk stayed at 89 %.

Gotchas:

- **The KFP flake had a bad day and shows more faces.** Of 6 submissions of the first run and 4 retries of the second, only 2 went through. The failures were always in KFP's own pods, before our code ran (a Python traceback is still a real failure): `dag-driver` and `container-driver` with `GetContextByTypeAndName ... Unavailable ... Error while dialing` or `dial tcp <metadata-grpc>:8080: i/o timeout`; `container-driver` with `Internal ... mysql_real_connect failed: errno: , error:` (the log of `metadata-grpc` says `Failed to connect to the database`); the KFP API answering 500 with `lookup mysql on 10.96.0.10:53: i/o timeout`; and the Argo `wait` container failing with `failed to put file: Get "http://minio-serv...`. The first run was rescued with a retry and the second only went through after restarting minikube, which may be luck and not the cause.
- **A probe pod measured it, and the cause was still not found.** Three fresh pods in `kubeflow` measured DNS (short name and FQDN) and TCP to the ClusterIPs of `mysql`, `metadata-grpc-service` and `ml-pipeline` for 150 s. CoreDNS (`log` is on) answered all its 2,566 queries in 6 ms at most. The first TCP connect to each ClusterIP timed out (5 s) in the first 15 s of every pod, and after that 25 stalls of 0.9 to 4.7 s hit the three pods **at the same moment** and with the same length, on DNS and on direct TCP alike, while no query reached CoreDNS. The node was not CPU-throttled (34 throttled periods in 11,969). So it is not CoreDNS and not the CPU quota, but the probe cannot tell a stall of the cluster network from a freeze of the node or of the Docker Desktop VM. A clock probe that does not use the network would tell them apart.
- **`MlflowClient` retries are not enough, but a run retry is.** The drivers of KFP have no `retryStrategy` in the Argo workflow. `Client(host=...)._run_api.run_service_retry_run(run_id)` (a private attribute of the SDK, and the KFP UI has the same Retry button) re-creates only the pods that failed and keeps the finished steps, so a run that fell at the Gate repeats Gate and Register in about 2 minutes and not the 5 of the whole training. It worked at the second try for the first run, and three retries of the second run failed in a row (two more were refused by the KFP API with a 500). The client of `submit_pipeline.py --wait` can also die with that 500 while the run goes on.
- **Failed attempts leave pods in `Error`.** Delete them (they are ours: `kubectl delete pod -n kubeflow <name>`) before the hygiene check above, and delete the probe pods after any network test.
- **`timeout` does not exist on macOS**, and `sh` in the `metadata-grpc` image has no `/dev/tcp` (a failed test with it was an artifact, not a network failure). That image has `bash` and `getent` but no `nc`, `curl` or `python3`.

## Deploy pipeline (SPEC 07, E3)

Pipeline 2 serves whatever is `churn@champion`: `fetch_champion` writes `models/churn/v<N>/model.joblib` on the PVC, `publish` creates or patches the InferenceService `churn` on that folder and waits until the new pod answers, and `smoke_test` checks that the endpoint returns the probabilities of that file. Each part is explained, with what was measured, in the subsections below.

**In order, from a stopped cluster** (each line is explained below; the commands were the ones actually run):

```bash
minikube start                                                    # then wait a few minutes for the pods to settle
kubectl apply -f k8s/rbac-pipeline-runner.yaml                    # once: Role + RoleBinding (step 3)
minikube image build -t churn-sklearnserver:0.1.0 docker/churn-sklearnserver   # once: the serving image (step 4)
kubectl apply -f k8s/serving-runtime.yaml                         # once: the ServingRuntime (step 5)
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888        # KFP API, in another terminal
uv run python tools/submit_pipeline.py --pipeline deploy --wait   # syncs churn-code, compiles, submits, waits up to 15 min
kubectl get isvc churn -n kubeflow                                # READY=True; the annotation churn/model-version says which version
kubectl port-forward -n kubeflow svc/churn-predictor 18080:80     # then POST {"instances": [[...10 raw values...]]} to :18080/v1/models/churn:predict
kubectl get pods -A --no-headers | awk '$4!="Running" && $4!="Completed"'   # hygiene: prints nothing (delete Error pods of failed attempts)
```

If the run fails inside a KFP pod with `dial tcp <metadata-grpc>:8080: i/o timeout`, it is the flake described under "Deploy runs on the cluster", not a bug in the components: retry the same run (and wait on the workflow, not on the API client).

### Constants and RBAC (steps 1 and 3)

Step 1 added seven constants to `components/config.py` (`PVC_CLAIM_NAME`, `K8S_NAMESPACE`, `ISVC_NAME`, `SERVING_RUNTIME_NAME`, `PUBLISH_TIMEOUT_S`, `SMOKE_ROWS`, `SMOKE_TOLERANCE`), which travel in the ConfigMap `churn-code`, so `tools/sync_code.sh` did not change. Step 3 applied the RBAC:

```bash
kubectl apply -f k8s/rbac-pipeline-runner.yaml
SA=system:serviceaccount:kubeflow:pipeline-runner
kubectl auth can-i get inferenceservices.serving.kserve.io -n kubeflow --as=$SA      # also create and patch: yes; delete: no
kubectl auth can-i get servingruntimes.serving.kserve.io -n kubeflow --as=$SA        # also list: yes; delete: no
```

A Role named `pipeline-runner` already existed in `kubeflow` (pods, PVCs, ConfigMaps, Argo workflows, `kubeflow.org`; nothing on KServe), so the new Role is called `pipeline-runner-serving` with the RoleBinding `pipeline-runner-serving-binding`, and the existing Role and binding were left alone. Before the apply, `can-i` printed `no` for everything on `serving.kserve.io`. The existing Role already lets `pipeline-runner` read Deployments and pods (`can-i get deployments.apps` and `get`/`list pods` print `yes`), which Publish uses to follow the rollout.

### Cluster assumptions checked before building anything (step 2)

Read on 2026-11-06 after `minikube start` (the cluster had been stopped since 2026-11-04). The start warned about low disk in the Docker VM, and the node showed `/var` at 89 % with 6.3 GB free, the same as before. `cpu.max` was still `800000 100000`. After about two minutes every deployment of `kube-system`, `cert-manager`, `kserve`, `kubeflow` and `mlflow` was `Available`, and no pod was outside `Running` or `Completed`. `minikube start` reports Kubernetes v1.32.0, not the v1.29 that README §4 says.

```bash
kubectl get rolebinding,clusterrolebinding -A -o json        # who is bound to pipeline-runner
kubectl get role pipeline-runner -n kubeflow -o yaml          # what that Role allows
kubectl auth can-i <verb> inferenceservices.serving.kserve.io -n kubeflow --as=system:serviceaccount:kubeflow:pipeline-runner
kubectl get clusterservingruntime kserve-sklearnserver -o yaml
kubectl run cb-k8s-check -n default --image=churn-base:0.3.0 --image-pull-policy=Never --restart=Never --rm -i --quiet --command -- python -c "import kubernetes; print(kubernetes.__version__)"
kubectl run ss-check -n default --image=kserve/sklearnserver:v0.21.0 --restart=Never --rm -i --quiet --command -- sh -c '<id, python, libgomp, versions>'
```

| Question | Answer | Evidence |
|---|---|---|
| Service account of the KFP step pods | **`pipeline-runner`**, confirmed in step 10 (it was first inferred, not observed) | The service account `pipeline-runner` exists in `kubeflow` and has a `RoleBinding` `pipeline-runner-binding` to a Role `pipeline-runner`. The `ml-pipeline` deployment sets no `DEFAULTPIPELINERUNNERSERVICEACCOUNT`, so KFP uses its built-in default. When this was checked no run pod was left to read `spec.serviceAccountName` from; in step 10 every pod of a deploy run showed `serviceAccountName: pipeline-runner` |
| What `pipeline-runner` may do today | Nothing on KServe | `can-i get`, `create`, `patch` and `delete` on `inferenceservices`, and `get` on `servingruntimes`, all printed `no`. The existing Role covers pods, PVCs, ConfigMaps, Argo workflows and `kubeflow.org`, not `serving.kserve.io` |
| Name of the Role for step 3 | A Role named `pipeline-runner` **already exists** in `kubeflow` | The new Role needs another name (`pipeline-runner-serving`, with the RoleBinding `pipeline-runner-serving-binding`), and the existing Role and binding stay untouched |
| `churn-base:0.3.0` imports `kubernetes` | yes | `kubernetes 30.1.0` (also `kfp 2.15.0`, Python 3.11.18) |
| Arguments of `kserve-sklearnserver` | `--model_name={{.Name}}`, `--model_dir=/mnt/models`, `--http_port=8080` | `protocolVersions` v1 and v2, `supportedModelFormats` `sklearn` version `1`, container `kserve-container`, `securityContext` with `runAsNonRoot: true`, `allowPrivilegeEscalation: false`, `drop: [ALL]`. Annotations `prometheus.kserve.io/path: /metrics` and `prometheus.kserve.io/port: "8080"` (useful for E4) |
| Resources of that runtime | requests and limits of **1 CPU and 2Gi** | The InferenceService of SPEC 07 sets its own `resources` (100m, 256Mi, limit 1Gi). They are **merged field by field**, not replaced (seen on the real pod in step 10): requests 100m and 256Mi, limit memory 1Gi, and the limit CPU `1` still comes from the runtime |
| User of `kserve/sklearnserver:v0.21.0` | `kserve`, uid 1000 (not root) | `id` in a throwaway pod |
| `pip` in the image | **No.** The venv `/prod_venv` (Python 3.11.16) has no `pip` and the image has no `uv` | `python -m pip` fails with `No module named pip`. `ensurepip` is available. The user `kserve` **can write** to `/prod_venv/lib/python3.11/site-packages` |
| `libgomp.so.1` in the image | **No system library.** Only a copy inside `scikit_learn.libs/libgomp-d22c30c5.so.1.0.0` | `ldconfig -p` has no match. `/usr/bin/apt-get` exists but needs root. Debian 12, `aarch64` |
| Versions in the image (baseline for the `pip freeze` check of step 4) | `scikit-learn 1.5.2`, `numpy 2.4.4`, `scipy 1.17.1`, `joblib 1.5.3`, `pandas 2.3.3`, `threadpoolctl 3.6.0`, `kserve 0.21.0` | 114 distributions in total. It has no `xgboost`, `lightgbm` or `cloudpickle` |

Gotchas:

- `kubectl get pods` listed the old `churn-training-*` pods in the first seconds after the start and none a few minutes later, so a service account cannot be read from a past run. A direct read needs a live step pod.
- A bare `kubectl run ... --rm` with `--image-pull-policy=Never` works for `churn-base` because `minikube image build` put it in the node. The `sklearnserver` image was already in the node from the spike, and `IfNotPresent` is enough.

### Derived serving image `churn-sklearnserver:0.1.0` (step 4)

`docker/churn-sklearnserver/Dockerfile` starts from `kserve/sklearnserver:v0.21.0`, installs `libgomp1` as root, installs the three packages of `requirements.txt` (`xgboost`, `lightgbm` and `narwhals`, which `lightgbm` requires) with `uv pip install --no-deps` (uv comes from `ghcr.io/astral-sh/uv:0.11.29` and is only bind-mounted for that `RUN`, so the final image has neither `uv` nor `pip`), and returns to uid 1000. The last `RUN` imports both libraries as that user, so a missing `libgomp` fails the build. The `uv` tag is the version installed on the laptop; if the registry has no such tag, the build fails at the first line and the tag is the thing to change.

```bash
minikube image build -t churn-sklearnserver:0.1.0 docker/churn-sklearnserver
minikube image ls | grep churn-sklearn
```

The check runs `tools/serving_image_check.py` in temporary pods built from `k8s/serving-image-check-pod.yaml` (a template: `sed` fills the name, image and mode). The script and the code reach the pods through the ConfigMaps `serving-image-check` and `churn-code`.

```bash
kubectl create configmap serving-image-check -n kubeflow \
  --from-file=serving_image_check.py=tools/serving_image_check.py --dry-run=client -o yaml | kubectl apply -f -
tools/sync_code.sh        # refreshes churn-code (config.py now has the SPEC 07 constants)

check() {   # check <pod name> <image> <arguments of serving_image_check.py as a YAML flow list body: fit | load | clean | query, --url, <url>, --family, <name>>
  sed -e "s|__NAME__|$1|" -e "s|__IMAGE__|$2|" -e "s|__ARGS__|$3|" k8s/serving-image-check-pod.yaml | kubectl apply -f -
  kubectl wait -n kubeflow pod/$1 --for=jsonpath='{.status.containerStatuses[0].state.terminated}' --timeout=240s
  kubectl logs -n kubeflow $1
  kubectl get pod -n kubeflow $1 -o jsonpath='exit code: {.status.containerStatuses[0].state.terminated.exitCode}{"\n"}'
  kubectl delete pod -n kubeflow $1 --wait=false
}
check image-check-fit  churn-base:0.3.0            fit     # writes models/image-check/ on the PVC
check image-check-load churn-sklearnserver:0.1.0   load    # the derived image: all five families ok
check image-check-orig kserve/sklearnserver:v0.21.0 load   # the original: xgboost and lightgbm fail
```

`models/image-check/` and the ConfigMap `serving-image-check` are temporary: they are removed at the end of step 5 (`check image-check-clean churn-base:0.3.0 clean`, then `kubectl delete configmap serving-image-check -n kubeflow`).

The script was also run on the laptop, outside the cluster, in a temporary folder: `fit` saved the five files (4.9 MB in total) and `load` printed `ok max_diff=0.000e+00` for the five families. With one file replaced by garbage, `load` printed the error for that family only and exited with code 1.

Observed (2026-11-06):

- **The first build failed at the import check, as designed.** With only `xgboost` and `lightgbm` in `requirements.txt`, the last `RUN` of the Dockerfile failed with `ModuleNotFoundError: No module named 'narwhals'`: `lightgbm 4.7.0` requires `narwhals>=1.15` and `--no-deps` skips it. `xgboost 3.2.0` needs only `numpy` and `scipy`. Adding `narwhals==2.28.0` (the `churn-base` pin) fixed it, and the second build printed `numpy 2.4.4 | scipy 1.17.1 | scikit-learn 1.5.2 | xgboost 3.2.0 | lightgbm 4.7.0` as uid 1000. The `uv` tag `0.11.29` exists in `ghcr.io`, and `RUN --mount=type=bind,from=uv` works with the builder of `minikube image build`. The install took about 13 s (the `xgboost` wheel is 125 MiB), and the `libgomp1` layer was cached on the second build.
- **The derived image loads the five families.** `fit` in a `churn-base:0.3.0` pod wrote the five files (exit 0). `load` in `churn-sklearnserver:0.1.0` (Python 3.11.16, numpy 2.4.4, scikit-learn 1.5.2) printed `ok max_diff=0.000e+00` for `logreg`, `random_forest`, `hist_gradient_boosting`, `xgboost` and `lightgbm`, exit 0. The models were pickled under numpy 2.4.6 and loaded under 2.4.4 with no difference at all.
- **The original image fails exactly where expected.** `load` in `kserve/sklearnserver:v0.21.0` passed `logreg`, `random_forest` and `hist_gradient_boosting` and printed `error ModuleNotFoundError: No module named 'xgboost'` and `... 'lightgbm'`, exit 1. So the derived image is what fixes it.
- **Package comparison.** `pip freeze` cannot run here (neither image has `pip`, by design), so the installed distributions were listed with `importlib.metadata` in both images and compared with `diff`. The original has 114 and the derived 117. The only differences are three added lines: `lightgbm==4.7.0`, `narwhals==2.28.0` and `xgboost==3.2.0`. `scikit-learn` is still `1.5.2`, `numpy` `2.4.4`, `scipy` `1.17.1`, `joblib` `1.5.3` and `pandas` `2.3.3`. Neither `pip` nor `uv` is in the derived image.
- **Disk.** The node `/var` went from 89 % (6.3 GB free) to 91 % (5.5 GB free) after pulling `uv`, the `xgboost` wheel and the new image layers.

### Serving runtime `churn-sklearnserver` (step 5)

`k8s/serving-runtime.yaml` is a namespaced `ServingRuntime` in `kubeflow` copied from the cluster's `kserve-sklearnserver` (same arguments, resources, `securityContext`, protocol versions and Prometheus annotations). Compared with `diff` on the two `spec` objects, the only difference is the image line. The cluster runtime is not edited.

```bash
kubectl apply -f k8s/serving-runtime.yaml
kubectl get servingruntime churn-sklearnserver -n kubeflow -o jsonpath='{.spec.containers[0].image}'   # churn-sklearnserver:0.1.0

# Temporary InferenceService on the xgboost file that step 4 left in models/image-check/
kubectl apply -f - <<'EOF'
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata: {name: churn-image-check, namespace: kubeflow}
spec:
  predictor:
    minReplicas: 1
    model:
      modelFormat: {name: sklearn}
      runtime: churn-sklearnserver
      storageUri: pvc://churn-data/models/image-check/xgboost
      env: [{name: PREDICT_PROBA, value: "true"}]
      resources: {requests: {cpu: 100m, memory: 256Mi}, limits: {memory: 1Gi}}
EOF
kubectl wait --for=condition=Ready isvc/churn-image-check -n kubeflow --timeout=300s

# The query runs inside the cluster, from a pod, against the service name (no port-forward)
kubectl create configmap serving-image-check -n kubeflow \
  --from-file=serving_image_check.py=tools/serving_image_check.py --dry-run=client -o yaml | kubectl apply -f -
check image-check-query churn-base:0.3.0 'query, --url, "http://churn-image-check-predictor.kubeflow.svc.cluster.local/v1/models/churn-image-check:predict", --family, xgboost'

# Cleanup
kubectl delete isvc churn-image-check -n kubeflow
check image-check-clean churn-base:0.3.0 clean
kubectl delete configmap serving-image-check -n kubeflow
```

Observed (2026-11-06):

- **The runtime serves an XGBoost model.** The InferenceService was `Ready` in 14 s. Its predictor pod ran `churn-sklearnserver:0.1.0` (container `kserve-container`), had no init containers, one extra volume `kserve-pvc-source` (the PVC, mounted by KServe from the `pvc://` URI) and 0 restarts. The query with 3 rows printed `QUERY status=200 shape=(3, 2) max_diff=0.000e+00 ok`: the answer equals the `predict_proba` that the training pod computed, to the last decimal.
- **Assumptions of the data model confirmed.** In Standard mode the InferenceService creates the service `churn-image-check-predictor` (ClusterIP, port 80), so the endpoint of SPEC 07 follows the pattern `http://<name>-predictor.kubeflow.svc.cluster.local/v1/models/<name>:predict`. `kubectl get isvc` shows a URL `http://<name>-kubeflow.example.com`, which is not reachable and is not the one to use.
- **Cleanup.** The InferenceService, its predictor pod, the folder `models/image-check/` (a `load` on it now fails with `FileNotFoundError`) and the ConfigMap `serving-image-check` are gone. The ServingRuntime stays. No pod is outside `Running` or `Completed`. The node disk is still 91 % (5.2 GB free).

Gotchas:

- **A pod does not see a ConfigMap change after it starts, and a new pod reads the current one.** After adding the mode `query`, the first query pod ran the old script (`invalid choice: 'query'`) because the ConfigMap `serving-image-check` had not been refreshed. Refresh it (the `kubectl create ... | kubectl apply` line above) after every edit of the script.
- **Quoting in `kubectl get -o custom-columns=...[0]...` fails in zsh** (`no matches found`), because the brackets are a glob. Quote the argument.

### Fetch champion component (step 6)

`components/fetch_champion.py` has no inputs. It reads the version behind `churn@champion` once (`get_model_version_by_alias`), loads `models:/churn/<N>` (by version), writes `models/churn/v<N>/model.joblib.tmp` and renames it to `model.joblib` with `os.replace`, reloads the final file on the first 3 rows of `processed/train.csv` and checks that `predict_proba` has shape `(3, 2)`. Outputs: `model_version`, `model_dir` and `storage_uri` (`pvc://churn-data/models/churn/v<N>`, with the path taken relative to the mount point of the PVC).

Local check (no cluster), the same method as Gate and Register: call `fetch_champion.python_func()` with `config.MLFLOW_TRACKING_URI` pointed at `sqlite:///<tmp>/mlflow.db` (an experiment created with `artifact_location` in the temporary folder), `config.MODELS_DIR_PVC` and `config.PROCESSED_DIR_PVC` pointed at the temporary folder, and a `train.csv` of 2,000 rows. The versions were registered with `serialization_format="cloudpickle"`, as Register does, one of `logreg` and one of `xgboost`.

Observed (2026-11-06), 12 checks, all passed:

- **One version:** the outputs name version `1`, `model_dir` ends in `models/churn/v1`, `storage_uri` is `pvc://churn-data/models/churn/v1`, and the folder holds only `model.joblib`.
- **Two versions, alias on 2 (an `xgboost` champion):** it writes `v2`, `v1` keeps the same SHA-256, and the only call to `mlflow.sklearn.load_model` was `models:/churn/2`, not `models:/churn@champion`.
- **Same version twice:** the second run replaces the file (newer modification time) and the folder holds nothing but `model.joblib` (no `.tmp` left).
- **No alias:** it raises a `RuntimeError` that names the alias and the model.
- **No `processed/train.csv`:** it raises `FileNotFoundError` and does not pass in silence.

### Publish component (step 7)

`components/publish.py` takes the `storage_uri` of Fetch and returns the `endpoint`. It takes the version number for the annotation from the end of the URI (`/v<N>`, and it raises on anything else). It checks that the `ServingRuntime` `churn-sklearnserver` exists, then reads the InferenceService `churn`: if it does not exist it calls `create` with the manifest of the spec, and if it does it calls `patch` with the same whole manifest (the Python client sends `application/merge-patch+json`, which replaces lists such as `env` whole, so a patch is the same desired state as a create). Then it polls every 5 s, up to `PUBLISH_TIMEOUT_S` (300 s), until the condition `Ready` is `True` **and** `status.observedGeneration` equals `metadata.generation` (step 10 showed that this is not enough and added a third condition, the Deployment rollout: see "Publish waits for the Deployment rollout" below). Both fields exist in the CRD of KServe 0.21 (`kubectl get crd inferenceservices.serving.kserve.io`, which also has `status.modelStatus`, not used here). The endpoint follows the service name seen in step 5: `http://churn-predictor.kubeflow.svc.cluster.local/v1/models/churn:predict`.

Local check (no cluster): call `publish.python_func(<uri>)` with `kubernetes.config.load_incluster_config` and `kubernetes.client.CustomObjectsApi` replaced by an in-memory stub (it raises `ApiException(404)` for missing objects, bumps `metadata.generation` when a patch changes the spec, and can return a stale status), and `time.sleep` and `time.monotonic` replaced so the loops run on a fake clock.

Observed (2026-11-06), 21 checks, all passed:

- **No InferenceService:** one `create`, no `patch`. The manifest has `runtime: churn-sklearnserver`, the `storageUri` received, `PREDICT_PROBA=true`, the annotation `churn/model-version`, `modelFormat` `sklearn`, `minReplicas` 1, and the resources of the spec (100m and 256Mi, limit 1Gi). It waited for the first reconcile before returning.
- **Existing InferenceService, new version:** one `patch`, no `create`, with the new `storageUri` and annotation. The stub reported `Ready` from the old revision for the first two reads (generation 2, `observedGeneration` 1), and Publish did not accept it: it slept twice and returned only when `observedGeneration` caught up. With the guard removed from a copy (`if ready:` instead of `if ready and seen:`), that check fails and nothing else does, so the check is not vacuous.
- **Same version again:** a patch that changes nothing leaves the generation as it is, and Publish returns without waiting.
- **Missing runtime:** it raises `ServingRuntime 'churn-sklearnserver' does not exist ...` and calls neither `create` nor `patch`.
- **Never Ready:** it raises after 300 s of the fake clock with the generation, the `observedGeneration` and the conditions in the message.
- **Forbidden (403):** a 403 on the runtime or on the InferenceService is raised as it is, and is not read as "does not exist" (which would have ended in a `create` that is then also forbidden).
- **`storage_uri` without `/v<N>`:** it raises `ValueError` before any API call.

What this does not prove: that a real KServe controller moves `observedGeneration` and `Ready` in this order after a `patch`. That is the first real contact with the API server and it happens in step 10.

### Smoke test component (step 8)

`components/smoke_test.py` takes the `endpoint` of Publish and the `model_dir` of Fetch and returns nothing: it passes by finishing and fails by raising. It reads the first 3 rows of `processed/train.csv` in `FEATURES` order (numbers as floats, strings as strings), posts `{"instances": [...]}` to the endpoint with `urllib` (standard library only; `grep -c "import requests\|import httpx"` prints 0), and then checks, in this order: status 200 (only 200, so a 201 fails too), a `predictions` list, shape `(3, 2)`, every value finite and in [0, 1], and finally that the largest difference with `predict_proba` of `model_dir/model.joblib` on the same rows is below `SMOKE_TOLERANCE` (1e-6). That last check is the one that fails for a stale revision: shape and range would pass for any model.

Local check (no cluster): call `smoke_test.python_func(<url>, <model dir>)` against a throwaway `http.server` on `127.0.0.1` whose answer the test chooses. The model is a `logreg` fitted on 2,000 rows, and the "truth" is computed from the JSON that the client sent, which is how the real server receives it.

Observed (2026-11-06), 16 checks, all passed:

- **A correct answer passes**, and the request is what a client sends: 3 rows of 10 values, `Geography` and `Gender` as strings, the other eight as floats, and the first row equal to the first row of `train.csv`. A difference of 5e-7 also passes (below the tolerance).
- **Status 500, 404 and 201 raise**, and the message carries the status and the start of the body.
- **Wrong shape raises**: 3 rows of 1 value, and 2 rows instead of 3. An answer without a `predictions` list raises with that cause. The first version of the component let `None` through `np.asarray` and said `shape ()`, which pointed at the wrong thing; it now checks for the list first.
- **A probability above 1 or below 0 raises**, with the values in the message.
- **A difference of 1e-5 raises**, and so does the answer of a different model (`[[0.9, 0.1]]` for every row), with "does not serve <model_dir>" in the message.
- **Nothing listening raises** `POST ... failed: [Errno 61] Connection refused`.

What this does not prove: the name resolution and the path from inside the cluster, which step 5 showed for a temporary service (`<name>-predictor`, port 80) and step 10 shows for `churn`.

### Deploy pipeline wiring (step 9)

`pipelines/deploy_pipeline.py` defines `churn-deploy`: `fetch_champion → publish(storage_uri) → smoke_test(endpoint, model_dir)`, with no parameters. Every task goes through `_configure`, imported from `pipelines/training_pipeline.py`, which is not touched: code ConfigMap `churn-code` at `/code`, PVC `churn-data` at `/mnt/data`, `PYTHONPATH=/code`, no caching, `MLFLOW_HTTP_REQUEST_MAX_RETRIES=10`, `OMP_NUM_THREADS=1`, `IfNotPresent`. `fetch_champion` asks 500m CPU and 512Mi with a 1.5Gi limit and the other two 100m and 256Mi with a 1Gi limit. `publish` does not read the PVC, which it gets only because `_configure` mounts it on every task.

`tools/submit_pipeline.py` takes `--pipeline training|deploy` (default `training`, so the old command is unchanged). A small table maps each name to its function, KFP experiment (`churn-training`, `churn-deploy`) and compiled package; the YAMLs under `pipelines/` are git-ignored.

```bash
uv run python tools/submit_pipeline.py --compile-only                      # training, as before
uv run python tools/submit_pipeline.py --pipeline deploy --compile-only    # writes pipelines/deploy_pipeline.yaml
uv run python tools/submit_pipeline.py --pipeline deploy --wait            # syncs churn-code, submits, waits up to 15 min
```

Observed (2026-11-06), read from the compiled YAML with `yaml.safe_load_all`:

- Both pipelines compile, and `--pipeline foo` is rejected by `argparse`. `git diff main -- pipelines/training_pipeline.py` is empty.
- Name `churn-deploy`, and the root has no `inputDefinitions`, so no pipeline parameters.
- Tasks `fetch-champion` (no dependency), `publish` (depends on `fetch-champion`, takes its `storage_uri`) and `smoke-test` (depends on both, takes `publish.endpoint` and `fetch-champion.model_dir`).
- The three executors use `churn-base:0.3.0`; the file contains no `0.2.0`.
- The three tasks mount `churn-data` at `/mnt/data` and `churn-code` at `/code`, and carry the three environment variables above. Their `cachingOptions` is `{}`, the same as the leaf tasks of the training pipeline.
- `tools/sync_code.sh` is unchanged and still builds `churn-code` with four keys.

### Deploy runs on the cluster (step 10, partial)

Commands: `kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888`, `uv run python tools/submit_pipeline.py --pipeline deploy --wait`. A failed run is retried with `Client(host=...)._run_api.run_service_retry_run(run_id)`. **Do not wait with `client.wait_for_run_completion`**: it died once with `500 ... lookup mysql ... no such host` while the run went on. Wait on the workflow itself instead: `until` its `.status.phase` is `Succeeded` or `Failed`, in a loop. (`kubectl wait --for=jsonpath='{.status.finishedAt}'` without a value returns at once, and `PIPESTATUS` does not exist in zsh.)

**The KFP flake, measured (2026-11-06).** 9 attempts end to end (submissions and retries): 1 reached `Succeeded`. Every failure had the same cause and the same address, `dial tcp 10.108.123.79:8080: i/o timeout`, which is `metadata-grpc-service`, and it happened in 5 different KFP pods before any of our code ran: `publish-driver` (4 times), `root-driver`, the driver of `smoke-test`, and the launcher of the `fetch-champion` executor (`failed to pre-publish Pod info to ML Metadata`, twice). The driver gives up after 20 s. `metadata-grpc` was `Running`, last restarted two hours before the first failure, and the node load was 0.45 on 12 CPUs. The cause is still not known. The pods of every run use the service account **`pipeline-runner`** (confirmed with `spec.serviceAccountName`).

**First run: `Succeeded`** (after retrying the same run). `fetch-champion` ran in the first attempt, `publish` created the InferenceService (`publish: created InferenceService kubeflow/churn -> pvc://churn-data/models/churn/v2`, then `Ready`), and `smoke_test` printed `smoke test ok: 3 rows, status 200, probabilities in [0, 1], max diff with /mnt/data/models/churn/v2/model.joblib 0.000e+00`. The InferenceService had `churn/model-version=2`, `uid 6519f4b2-2a3e-4286-a18c-4479f2075584`, generation 1, and `/mnt/models` held one file, `model.joblib` (195 KB).

**The predictor pod** (`kubectl get pod ... -o json`): image `churn-sklearnserver:0.1.0`, `args` `--model_name=churn --model_dir=/mnt/models --http_port=8080` (the `{{.Name}}` of the runtime became `churn`), env `PREDICT_PROBA=true` plus `INFERENCE_SERVICE_NAME=churn`, and a single mount `/mnt/models` from the volume `kserve-pvc-source` (claim `churn-data`) with **`subPath: models/churn/v2`**, read only. The service is `churn-predictor`, ClusterIP, port 80.

**The patch, tested outside KFP.** `publish` and `smoke_test` were run as the real components in a pod with the service account `pipeline-runner` (a temporary ConfigMap with the two files and a watcher that logged every change of `generation`, `observedGeneration`, `Ready` and the predictor pods every 0.5 s). Versions 1 and 2 of the model give **identical probabilities** (max difference 0.0 on 50 rows: same seed, params and data), so the smoke test cannot tell them apart in this experiment.

- **Patch that changes nothing (v2 on v2):** `patched`, back in 0.3 s, generation stays 1, and the smoke test passes with difference 0.
- **Rollout 1, v2 to v1:** generation 1 to 2 at t=5.3 s. `observedGeneration` stayed 1 and `Ready` stayed `True` (from the old revision) until t=22.0 s: **16.7 s of `Ready=True` with a stale `observedGeneration`**. The new pod (`...8xdfm`, subPath v1) went `Pending` at 6.9 s and `Running` at 7.9 s, and the old pod terminated at 18.3 to 19.9 s. `publish` returned at 21 s, after the old pod was gone. Without the `observedGeneration` check it would have returned at about 5 s with the old pod serving. Right after, the smoke test failed with `Temporary failure in name resolution` for `churn-predictor.kubeflow.svc.cluster.local` (a fresh pod resolved it fine 40 s later, in 0.01 s to 0.84 s; one sample, cause unknown).
- **Rollout 2, v1 to v2 (the restore):** generation 2 to 3 at t=1.6 s, **`observedGeneration` caught up at t=2.7 s with `Ready=True` while the new pod (`...x8g4d`) was still `Pending` and the old pod was the only one serving**. The new pod became `Running` at 3.8 s and the old one started terminating at 14.9 s. `publish` returned at 5.3 s only because it polls every 5 s; a poll at 3 s would have returned with the old pod serving. A DNS probe every 0.5 s saw no failure in this rollout.
- **Conclusion:** `Ready=True` with `observedGeneration == generation` does **not** mean the new revision is the one answering (rollout 2), and the time the controller takes to update `observedGeneration` varies (about 17 s in rollout 1, about 1 s in rollout 2). This is the case that step 10 of SPEC 07 says to stop on. What replaced it, and the measurement that settled it, is in the next subsection.
- The experiment left `churn` serving v1 for a few minutes because the script stopped at the failed smoke test; a second pod restored v2 (`churn/model-version=2`, same `uid`, generation 3). The folder `models/churn/v1/` created for the experiment was removed. No temporary pod or ConfigMap is left.

### Publish waits for the Deployment rollout (step 10)

Decision of 2026-11-06 (option 1 of three): after the two rollouts above, `publish` also waits for the rollout of the predictor Deployment `churn-predictor` to be complete. The service account `pipeline-runner` already reads Deployments and pods through its existing Role (`can-i get deployments.apps` and `get`/`list pods` print `yes`), so `k8s/rbac-pipeline-runner.yaml` did not change.

The condition, evaluated on every poll together with the two older ones: the Deployment `observedGeneration` is at least its `metadata.generation`, and `status.replicas`, `status.updatedReplicas` and `status.availableReplicas` all equal `spec.replicas`. A pod that is terminating is not counted in `status.replicas`, so the condition is true only after the old pod stopped receiving traffic. A missing Deployment keeps it waiting and, at the timeout, the message says `Deployment churn-predictor not found`.

Measured with a watcher (every 0.5 s: ISVC `generation`, `observedGeneration`, `Ready`, `modelStatus`, Deployment counters, and each pod's `subPath`, phase and `Ready`), two rollouts per version of the code, always ending with a `finally` that deploys v2 again:

All times are seconds after the `patch` (the clock of the script minus the moment `publish` started):

| Rollout | `publish` before (Ready + `observedGeneration`) returned at | New pod Ready (old pod starts terminating) | `publish` now (+ Deployment) returned at | Condition true at |
|---|---|---|---|---|
| v2 to v1 | 5.5 s, with the new pod `Running` but not Ready | 13.4 s (before) and 13.6 s (now) | 15.7 s | 13.6 s |
| v1 to v2 | 5.3 s, with the new pod `Running` but not Ready | 15.6 s (before) and 13.3 s (now) | 15.9 s | 13.8 s |

- Before the change `publish` returned about 5 s after the patch, while only the old pod could answer and the new pod still needed 8 to 10 s to be Ready, in both rollouts. In the first experiment of step 10 it looked correct (it waited 21 s) only because the controller was slow that time to update `observedGeneration`.
- After the change it returned 2.1 s and 2.1 s after the condition became true (the poll is every 5 s), and never before it. The condition was `False` during the whole window in which the old pod was the only one Ready. Two rollouts is a small sample.
- `status.modelStatus.transitionStatus` stayed `UpToDate` for the whole rollout in all four rollouts measured, so it is not a usable signal.
- A new pod takes about 8 to 10 s from `Running` to Ready, and the old one is deleted only then, so a model change takes about 13 to 16 s from the patch until the new pod answers, and `publish` returns about 16 s after the patch.
- The measurement ran in a pod with the service account `pipeline-runner` outside KFP, using the real `publish` component (a temporary ConfigMap with the file, removed afterwards). The final state is the champion: `churn/model-version=2`, same `uid`, one predictor pod, and no temporary object.

### Step 10 close-out

- **`curl` from the laptop** (`kubectl port-forward -n kubeflow svc/churn-predictor 18080:80`, port 18080 because the KFP UI may hold 8080): one raw row in `FEATURES` order (`675.0, "France", "Female", 37.0, 5.0, 0.0, 1.0, 1.0, 0.0, 131778.57`) answered `HTTP 200` with `{"predictions":[[0.5072928468077883,0.4927071531922117]]}`. Both values are in [0, 1], and they equal `predict_proba` of `models:/churn@champion` loaded from MLflow on the laptop, with a maximum difference of 0.
- **One run in KFP reached `Succeeded`, and it used the first version of `publish`** (Ready + `observedGeneration`). The version that also waits for the Deployment rollout was verified in the cluster outside KFP, with the same image, the same service account and the same component code, but it has **not** run inside a KFP pod.
- **The second run in KFP was not done, by decision of the user** (eight of the nine attempts failed with the same KFP flake, see above, and the patch path had already been exercised outside KFP). So the criterion "the second deploy run finishes `Succeeded` and keeps the same `uid`" is **not met as written**. What exists instead: the patch of `publish` against the real InferenceService ran nine times in a pod (three with the same spec, and six changing between v1 and v2), always with the same `uid` (`6519f4b2-2a3e-4286-a18c-4479f2075584`) and no recreation.
- **Times.** The workflow of the first run (`churn-deploy-4cdmp`) took 9 min 22 s from the first submission to `Succeeded`, with three retries of the same run in between, so it says little about the cost of a clean run. Inside the cluster, from the logs of that run: the `fetch_champion` pod started at 03:02:52Z and logged `publish success` at 03:03:00Z (about 8 s), and a model change takes about 16 s from the patch to `publish` returning. A clean-run time was not measured.
- **Hygiene.** The Error pods left by failed attempts were deleted. No pod is outside `Running` or `Completed`, no temporary pod or ConfigMap is left, and the node disk is at 91 % (5.2 GB free). The port-forwards to the KFP API (8888) and MLflow (5001) were still open at this point.

## Monitoring (SPEC 08, E4.1 to E4.3)

### E4.1 Serving metrics (measured 2026-11-07, no code needed)

- `GET /metrics` on port 8080 of the predictor pod answers `200`.
- It exposes the histograms `request_preprocess_seconds`, `request_predict_seconds`, `request_postprocess_seconds` and `request_explain_seconds`, all labelled `model_name="churn"`. The custom-predictor fallback of README §7 is not needed.
- **There is no error metric.** No metric carries an HTTP status, so an error-rate panel cannot be built from the predictor. SPEC 09 decides how to cover it.
- The container declares no named port and the pod annotations `prometheus.kserve.io/*` are not read (`enable-prometheus-scraping` is `"false"`). The scrape therefore goes through the Service `churn-predictor` (named port, 80 to 8080) with a `ServiceMonitor`.

### Values file and dry render (step 1)

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm template monitoring prometheus-community/kube-prometheus-stack --version 92.22.0 \
  -n monitoring -f k8s/monitoring-values.yaml > render.yaml
```

Result (Helm 4.3.1, no cluster needed): exit 0. The output has no Alertmanager, node-exporter or kube-state-metrics workloads. The `Prometheus` object has `retention: "24h"`, a 2Gi `volumeClaimTemplate` with class `standard` and empty `serviceMonitorSelector` / `podMonitorSelector` (the `*SelectorNilUsesHelmValues: false` effect). The Grafana Deployment has no `persistentVolumeClaim`.

Gotchas:

- The render still provisions a Grafana datasource named "Alertmanager" (it points at a service that does not exist) and keeps the operator RBAC for `alertmanagers`. Harmless; ignore the red datasource in Grafana.
- The first render still had Grafana (before it moved out of the cluster), and its generated Secret was the only place with a password. With `grafana.enabled: false` the render has no Grafana object and no password at all.

### Gates before the install (step 1, measured 2026-11-09 after `minikube start`)

| Measure | Command | Before the install |
|---|---|---|
| Node memory | `minikube ssh -- free -m` | 7834 total, 4810 used, 3023 available |
| Node disk | `minikube ssh -- df -h /var` | 59 GB, 39 GB used, 17 GB free (70 %) |

Gates of the spec (at least 1.5 GB of available memory and 4 GB of free disk): both pass.

Gotcha: the disk of minikube is the disk of the Docker VM. The minikube volume (about 28 GB, of which 25 GB is `/var/lib/containerd`) shares it with the host images and the build cache, so `df` inside minikube moves when the host is pruned. Right after `minikube start` the disk was at 91 % (5.2 GB free); `docker builder prune -f` on the host (4.5 GB of build cache) took it to 70 % (17 GB free). Memory right after a start is not representative: pods are still settling, so measure again a few minutes later.

### Install of the chart (step 2, 2026-11-09)

Final command (chart 92.22.0, Helm 4.3.1):

```bash
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --version 92.22.0 -n monitoring --create-namespace \
  -f k8s/monitoring-values.yaml --wait --timeout 10m
```

Result: release `monitoring` `deployed` (revision 2), 13 s on the last run. Pods `monitoring-kube-prometheus-operator` and `prometheus-monitoring-kube-prometheus-prometheus-0` are `Running` and ready, the PVC of Prometheus is `Bound` (2Gi), 10 CRDs of `monitoring.coreos.com`, no `OOMKilled`. Node after: 2634 MB available, 15 GB free on `/var` (74 %), load average 11.9.

What happened before (stop rule of the spec, no blind retries):

- **First install, Grafana inside the chart.** `--wait` failed after 10 min: `Deployment/monitoring-grafana not ready`. The container `grafana` (Grafana 13) never opened port 3000; the kubelet killed it by liveness (exit 137, not `OOMKilled`) and it restarted 3 times in 10 min.
- **Second try, liveness `initialDelaySeconds: 300`.** Same result after 11 min 24 s, and now `helm` and `kubectl` timed out with `TLS handshake timeout`. Inside the node the load average was 254 on 12 CPUs (Grafana, the operator, the KServe controller and the API server at 25 to 46 % CPU each). Memory was not the cause (about 1.9 GB available).
- **Decision:** Grafana runs outside the cluster, in a container on the Mac (`docker/grafana/compose.yaml`, SPEC 08 decision), and `grafana.enabled: false`.

Gotchas:

- **A failed Helm upgrade can leave the release in `pending-upgrade`.** The next `helm upgrade` then says `another operation (install/upgrade/rollback) is in progress`. The status lives in a Secret: `kubectl get secret -n monitoring -l owner=helm -L status` shows it, and `kubectl delete secret -n monitoring sh.helm.release.v1.monitoring.v<N>` for the `pending-upgrade` revision unlocks it. The release stays `failed` and the next upgrade works.
- **`helm list` in Helm 4 has no `--all` flag.** Use `helm list -n monitoring` or `helm history monitoring -n monitoring`.
- **Restarts during the incident are not a regression of the stack.** The operator shows 7 restarts and Prometheus 3, all during the minutes of overload caused by Grafana. The restart counts are checked again after the settle time of step 6.
- Under load the API server answers `TLS handshake timeout` in bursts; a command that fails once can work 20 s later. Wait for `minikube ssh -- uptime` to drop below about 20 before changing anything.

### Prometheus and Grafana answer (step 3, 2026-11-09)

```bash
kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 19090:9090   # background
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:19090/-/ready                    # 200
# query: up{job="monitoring-kube-prometheus-prometheus"}                                   # 1 (endpoints http-web and reloader-web)
docker compose -f docker/grafana/compose.yaml up -d                                        # run by the user
curl -s http://localhost:3000/api/health                                                   # "database": "ok", version 13.3.0
```

Grafana datasource (created by hand in the UI, Connections, Data sources, Prometheus): URL `http://host.docker.internal:19090`. Save & test answers "Successfully queried the Prometheus API". The first login (`admin` / `admin`) asks for a new password, which is not stored anywhere in the repo.

To reopen after a restart: start minikube, wait for `minikube ssh -- uptime` to drop below about 20, reopen the port-forward above, then `docker compose -f docker/grafana/compose.yaml up -d`. The datasource persists in the volume `grafana-data`; Grafana only shows data while the port-forward is open.

### Scrape of the predictor (step 4, 2026-11-09)

```bash
kubectl apply -f k8s/servicemonitor-churn.yaml
kubectl apply -f k8s/networkpolicy-allow-monitoring.yaml
# Prometheus: GET /api/v1/targets -> job="churn-predictor", namespace="kubeflow", health="up"
```

The `ServiceMonitor` alone left the target `down` (`Get "http://10.244.0.30:8080/metrics": context deadline exceeded`, scrape timeout of 10 s) even though the labels and the port name matched and `/metrics` answered in 38 ms through a port-forward.

Gotcha: **the namespace `kubeflow` has a `NetworkPolicy` `default-allow-same-namespace`** (`podSelector: {}`, ingress only from the same namespace), enforced by kindnet. A pod in another namespace, like Prometheus in `monitoring`, is dropped silently: no error, only a timeout. `k8s/networkpolicy-allow-monitoring.yaml` adds a second policy that opens TCP 8080 of the predictor pods (`component=predictor`) to the namespace `monitoring`. Policies add up, so the existing one is untouched. After applying it the target was `up` on the first check. To diagnose this again: compare the time of `/metrics` through a port-forward (it goes through the node, not the pod network) with the scrape error, and run `kubectl get networkpolicy -A`.

### Final measurement after 10 minutes (step 6, 2026-11-09)

Taken 10 minutes after the check of step 5, with the target `churn-predictor` scraping every 15 s and no other traffic.

| Measure | Before the install | After 10 minutes |
|---|---|---|
| Node memory (`minikube ssh -- free -m`) | 7834 total, 4810 used, 3023 available | 5510 used, 2323 available |
| Node disk (`minikube ssh -- df -h /var`) | 17 GB free (70 %) | 14 GB free (77 %) |
| Load average (1 / 5 / 15 min) | 254 at the worst (Grafana inside the cluster) | 0.32 / 1.32 / 20.0 (the 15 min value still holds the incident) |
| Restarts in `monitoring` | none yet | operator 7, Prometheus 3: the same as before the 10 minutes, so no new restart |
| Target `up{job="churn-predictor"}` | n/a | 1 |
| Data of Prometheus | n/a | about 1.1 MB (WAL 976 KB, head chunks 128 KB, 1846 series) |

Gotchas:

- `prometheus_tsdb_storage_blocks_bytes` is `0` in the first hours. Prometheus keeps the recent data in the WAL and the head, and writes the first block only after about 2 h. To size the data early, add `prometheus_tsdb_wal_storage_size_bytes` and `prometheus_tsdb_head_chunks_storage_size_bytes`.
- The Prometheus image has no `du` or `df`, so `kubectl exec ... du -sm /prometheus` fails. Use the metrics above.
- The restarts counted (operator 7, Prometheus 3) are from the Grafana overload of step 2, they did not grow afterwards.
- The disk went from 17 GB to 14 GB free after the install: the images of the stack plus the PVC.

## Load generator, dashboard and submission (SPEC 09, E4.4 to E4.6)

### Assumptions checked on the cluster (step 1, 2026-11-09)

Checked with the helper pod `k8s/data-loader-pod.yaml` and a temporary pod with `churn-base:0.3.0` that mounted `churn-data` and `churn-code`; both pods were deleted afterwards.

- **`test.csv` is at `/mnt/data/raw/test.csv`** (`RAW_DIR_PVC`): 10,001 lines, that is 10,000 rows plus the header, with every column of `FEATURES`. `sample_submission.csv` and `train.csv` are in the same folder.
- **`/mnt/data/submission.csv` does not exist yet.** The directory `/mnt/data` is `drwxrwxrwx` and `churn-base` has no `USER` line, so the pods run as root and can write there.
- **`pandas` 3.1.0 imports** in `churn-base:0.3.0`.
- **The endpoint answers from a pod in `kubeflow`:** `POST http://churn-predictor.kubeflow.svc.cluster.local/v1/models/churn:predict` with the first row of `test.csv` (numbers as floats, strings as strings) returned `HTTP 200` and `{"predictions":[[0.7604308402435738,0.23956915975642623]]}`. No port-forward is needed from inside the cluster.
- **That request counts in the predictor.** It raised `request_predict_seconds_count` by 1, so the baseline of the `+100` check of step 4 is taken right before the Job, not from this step.
- Node before the step: load average 0.35, 2298 MB available, 14 GB free on `/var`.

### Load generator, local checks (step 2, 2026-11-09)

`tools/load_generator.py` was checked on the laptop against a throwaway HTTP server (a Python `ThreadingHTTPServer` that answers `p1 = CreditScore / 1000`, so the answer depends on the row and the order of the answers can be proven). The 250 rows are the first rows of `data/test.csv`. The check script is not in the repo; it covers:

- one pass over 250 rows sends 3 requests of 100, 100 and 50 rows, each row has the 10 features in `FEATURES` order, numbers as floats and strings as strings;
- the file has `id,Exited`, the ids in the input order and `Exited` equal to the `p1` of that row;
- 25 requests at 10 per second arrive with a mean gap of 0.1 s (within 10 %);
- a server that fails twice and then answers: 2 `error` and 3 `ok` attempts, and the file is written;
- a server that always fails: exit code 1 after 4 attempts of the first batch, no file, and a stale file that was there is gone;
- a wrong number of rows, a value of 1.5, a `NaN` and a body that is not JSON each count as exactly one error and fail the run;
- `GET /metrics` returns `load_requests_total{result="ok"}` and `{result="error"}` at 0 before any request, follows the counters, and answers 404 elsewhere;
- 2 passes send 6 requests, and the file is still the one of the first pass;
- the script imports only the standard library and `config`.

Result: 24 of 24 checks passed. To run the generator locally: `PYTHONPATH=components:tools uv run python tools/load_generator.py --url <endpoint> --data data/test.csv --output /tmp/submission.csv --metrics-port 0`.

### Submission check, local (step 3, 2026-11-09)

```bash
uv run python tools/check_submission.py <submission.csv> [--sample data/sample_submission.csv]
```

Exits 0 and prints one `RESULT ok` line only if the header is exactly `id,Exited`, there are 10,000 rows, the ids equal those of the sample file in the same order, and every `Exited` is a number in [0, 1]. Otherwise it prints `RESULT fail: <first violation>` and exits 1. Standard library only, so it needs no `PYTHONPATH`.

Checked with `data/sample_submission.csv` (passes) and with copies written to a temporary folder (the check script is not in the repo). 17 of 17 cases gave the expected result:

- **Valid:** the sample file, a file with 10,000 random probabilities, and a file with the boundary values 0 and 1.
- **The six cases of the spec, each exit 1 naming the violation:** a missing row (`there are 9999 rows`), a changed id (`row 5 has id '99999', the sample has '15004'`), two ids swapped (the first of the two rows), a `NaN`, a value of 1.2, and a wrong header.
- **More, each exit 1:** an extra row, text instead of a number, a negative value, an empty value, `inf`, a row with one column, a file that does not exist and an empty file.

Gotcha: the check proves format, ids and range, not that row N got the prediction of row N. The order of the answers is proven by the local test of the generator (step 2), and `0.5` everywhere, as in the sample file, also passes.
