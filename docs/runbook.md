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
