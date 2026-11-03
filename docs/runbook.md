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
uv export --no-dev --no-hashes --format requirements-txt > docker/churn-base/requirements.txt
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
uv export --no-dev --no-hashes --format requirements-txt > docker/churn-base/requirements.txt
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
