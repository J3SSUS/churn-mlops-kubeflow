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
