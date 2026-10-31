# SPEC 03 — MLflow in the cluster

> **Status:** Approved
> **Depends on:** SPEC 01, SPEC 02
> **Backlog:** E1 (MLflow in the cluster), tasks E1.1–E1.4 of `BACKLOG.md`
> **Date:** 2026-10-31
> **Objective:** Deploy an MLflow tracking server with model registry in the `mlflow` namespace and verify that a `churn-base` pod in `kubeflow` can log a run with an artifact and register, alias and load back a test model.

## Why this spec exists

E1 fits in one spec: four small tasks on one component, with one "done when".

Two things are not obvious:

- The server and the client live in different images. The server uses the official image with the full `mlflow` package. The client is `mlflow-skinny` inside `churn-base`. Both are pinned to `3.17.2`.
- The checks leave the server installed and clean up only their own test objects, because E2 builds on it.

## Scope

**In:**

- **E1.1 Manifests:** `k8s/mlflow.yaml` with four objects.
  - Namespace `mlflow`.
  - PVC `mlflow-data`: 2Gi, `ReadWriteOnce`, storageClass `standard`.
  - Deployment `mlflow`: image `ghcr.io/mlflow/mlflow:v3.17.2`, 1 replica, `strategy: Recreate`, small resource requests and limits, a readiness probe on `/health`. It runs `mlflow server --serve-artifacts --workers 1` with SQLite and the artifacts on the PVC, and `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`.
  - Service `mlflow` on port 5000.
- **E1.2 Deploy and UI:** apply, wait for `Available`, `kubectl port-forward` to a free local port (5001) and open the UI.
- **E1.3 Connectivity test:** a pod in `kubeflow` with `churn-base:0.1.0` logs a run with a param, a metric and one small artifact to `http://mlflow.mlflow.svc.cluster.local:5000`, in the experiment `connectivity-test`.
- **E1.4 Registry test:** the same pod registers a dummy sklearn model as `registry-test`, sets an alias and loads it back with `models:/registry-test@<alias>`.
- **Test script:** `tools/mlflow_check.py` with three modes (`connectivity`, `registry`, `cleanup`). It reaches the pod through a temporary ConfigMap, as in SPEC 02. The two test modes leave their objects in place, so the run can be seen in the UI before the cleanup.
- **Cleanup of the test objects:** delete the registered model, the runs and the experiment through the API, and remove the temporary pod and ConfigMap. MLflow keeps the deleted runs and experiment as soft-deleted rows (no `mlflow gc`), so the name `connectivity-test` cannot be reused. The server, its PVC and its Service stay, because E2 needs them.
- **Docs:** a "MLflow" section in `docs/runbook.md`, a line in the `DECISIONS.md` log with the pinned MLflow version, and the README status of MLflow (§4) updated from "to install" to installed.

**Out of scope (for future specs):**

- The MLflow constants (tracking URI, experiment name, model name) in `components/config.py`, next to the rest of the shared configuration. They are fixed values, unlike run ids, which MLflow generates on every run. They land in E2.1.
- Registering the real `churn` model or setting `@champion` (E2.7).
- A custom MLflow server image. If the official image has no arm64 variant, the fix gets its own decision first.
- Postgres or any backend other than SQLite (README fallback, only if locks appear in E2).
- Authentication, Ingress, backups, and MLflow metrics in Prometheus.

## Data model

This spec introduces no data schema of its own. It defines the cluster objects, the MLflow storage layout and the names used by the checks.

### Cluster objects

| Object | Namespace | Notes |
|---|---|---|
| Namespace `mlflow` | — | New |
| PVC `mlflow-data` | `mlflow` | 2Gi, `ReadWriteOnce`, storageClass `standard` |
| Deployment `mlflow` | `mlflow` | Image `ghcr.io/mlflow/mlflow:v3.17.2`, 1 replica, `strategy: Recreate` (RWO volume) |
| Service `mlflow` | `mlflow` | ClusterIP, port 5000 |
| Pod `mlflow-check` | `kubeflow` | `churn-base:0.1.0`, temporary, from `k8s/mlflow-check-pod.yaml` |
| ConfigMap `mlflow-check-code` | `kubeflow` | Holds `tools/mlflow_check.py`, temporary |

### Server command

```text
mlflow server --host 0.0.0.0 --port 5000
  --backend-store-uri sqlite:////mlflow/mlflow.db
  --artifacts-destination /mlflow/artifacts
  --serve-artifacts
  --workers 1
  --allowed-hosts mlflow.mlflow.svc.cluster.local:5000,mlflow.mlflow.svc.cluster.local,localhost:*,127.0.0.1:*
```

The Deployment also sets the env var `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`.

- The client only needs `MLFLOW_TRACKING_URI`. Artifacts travel over HTTP, so pods in `kubeflow` need no access to the `mlflow-data` volume.
- The client in `churn-base` is `mlflow-skinny 3.17.2`, the same version as the server.

### MLflow PVC layout (mounted at `/mlflow`)

```text
/mlflow/
├── mlflow.db            # SQLite backend store
└── artifacts/           # artifact store, served by the tracking server
```

### Endpoints and names

| Item | Value |
|---|---|
| In-cluster URL | `http://mlflow.mlflow.svc.cluster.local:5000` |
| Local UI | `kubectl port-forward -n mlflow svc/mlflow 5001:5000` → `http://localhost:5001` |
| Test experiment (E1.3) | `connectivity-test` |
| Test model (E1.4) | `registry-test` (never `churn`) |
| Test alias | `check` (never `champion`) |

### `tools/mlflow_check.py` (modes)

| Mode | Does | Prints |
|---|---|---|
| `connectivity` | Logs one run in `connectivity-test` with a param, a metric and a small text artifact, then reads the artifact back | Run id, artifact URI |
| `registry` | Fits a `DummyClassifier(strategy="prior")` on a tiny array, logs it with signature and input example, registers it as `registry-test`, sets the alias `check`, loads `models:/registry-test@check` and compares `predict_proba` with the local model | Model version, max absolute difference |
| `cleanup` | Deletes the registered model `registry-test`, then the runs and the experiment `connectivity-test` | What was deleted |

The pod `mlflow-check` runs `sleep`, and each mode is a `kubectl exec ... python /code/mlflow_check.py <mode>`, so one pod serves the three steps.

## Implementation plan

1. **Decision line and manifest (E1.1).**
   - Add one line to the `DECISIONS.md` log: MLflow server pinned to `3.17.2` (same as the `mlflow-skinny` client in `churn-base`), official image `ghcr.io/mlflow/mlflow`.
   - Write `k8s/mlflow.yaml` (data model): Namespace, PVC, Deployment, Service.
   - Check: `kubectl apply --dry-run=client -f k8s/mlflow.yaml` exits 0.
2. **Deploy and UI (E1.2).**
   - `kubectl apply -f k8s/mlflow.yaml` and `kubectl wait --for=condition=Available deploy/mlflow -n mlflow`.
   - `kubectl port-forward -n mlflow svc/mlflow 5001:5000` and open `http://localhost:5001`.
   - Check: the pod is `Running`, `curl http://localhost:5001/health` answers `OK`, and the UI loads in the browser.
   - If the pod fails with `ImagePullBackOff` or `exec format error`, save the events as evidence and stop: the custom-image fallback needs its own decision.
3. **Connectivity test (E1.3).**
   - Write `tools/mlflow_check.py` with the `connectivity` mode only, and `k8s/mlflow-check-pod.yaml` (`churn-base:0.1.0`, `IfNotPresent`, `sleep`, `MLFLOW_TRACKING_URI` set, ConfigMap mounted at `/code`).
   - Create the ConfigMap `mlflow-check-code` and apply the pod in `kubeflow`.
   - `kubectl exec -n kubeflow mlflow-check -- python /code/mlflow_check.py connectivity`.
   - Check: the script prints a run id and reads the artifact back, and the run shows in the UI under `connectivity-test` with its param, metric and artifact.
4. **Registry test (E1.4).**
   - Add the `registry` mode to `tools/mlflow_check.py`.
   - Recreate the ConfigMap and the pod (pods are immutable and the old code would stay).
   - Run the `registry` mode.
   - Check: it prints model version 1 of `registry-test` and a maximum absolute difference below `1e-6`, and the model with alias `check` shows in the UI under Models.
5. **Cleanup of the test objects.**
   - Add the `cleanup` mode, recreate the ConfigMap and the pod, and run it.
   - Delete the pod `mlflow-check` and the ConfigMap `mlflow-check-code`.
   - Check: the registry has no `registry-test`, `connectivity-test` shows as deleted, `kubectl get pod,cm -n kubeflow` lists no `mlflow-check*` objects, and `kubectl get pods -n mlflow` still shows the server `Running`.
6. **Docs.**
   - Write the "MLflow" section of `docs/runbook.md` with the commands actually run, the observed results and the gotchas.
   - Update README §4: MLflow server and PVC from "to install" to installed.
   - Check: every command in the runbook section appears in the steps above, and README §4 no longer says "to install" for MLflow.

## Acceptance criteria

### MLflow server (E1.1, E1.2)
- [ ] `k8s/mlflow.yaml` defines Namespace `mlflow`, PVC `mlflow-data` (2Gi, `ReadWriteOnce`, `standard`), Deployment `mlflow` (image `ghcr.io/mlflow/mlflow:v3.17.2`, `strategy: Recreate`) and Service `mlflow` on port 5000.
- [ ] The Deployment command uses `--backend-store-uri sqlite:////mlflow/mlflow.db`, `--artifacts-destination /mlflow/artifacts` and `--serve-artifacts`.
- [ ] `kubectl get pvc mlflow-data -n mlflow` shows `Bound` with 2Gi.
- [ ] `kubectl get deploy mlflow -n mlflow` shows `1/1` available, and the pod has no restart loop.
- [ ] With the port-forward open, `curl http://localhost:5001/health` answers `OK` and the UI loads in the browser.
- [ ] `kubectl get pods -A --no-headers | awk '$4!="Running" && $4!="Completed"'` prints nothing, so the cluster stays healthy with MLflow added.

### Connectivity test (E1.3)
- [ ] The pod `mlflow-check` ran image `churn-base:0.1.0` with `MLFLOW_TRACKING_URI=http://mlflow.mlflow.svc.cluster.local:5000`.
- [ ] `python /code/mlflow_check.py connectivity` exits 0, prints a run id and reads the artifact back with the same content it logged.
- [ ] The run shows in the UI under `connectivity-test` with its param, metric and artifact. The check is recorded in the runbook.
- [ ] `kubectl exec -n mlflow deploy/mlflow -- ls /mlflow/artifacts` lists the experiment's artifact folder, which proves the artifact went through the server and not through a client-side write.

### Registry test (E1.4)
- [ ] `python /code/mlflow_check.py registry` exits 0, loads `models:/registry-test@check` and prints version 1 and a maximum absolute difference below `1e-6`.
- [ ] Before the cleanup, the UI lists `registry-test` under Models with the alias `check` on version 1. The check is recorded in the runbook.

### Cleanup
- [ ] After `python /code/mlflow_check.py cleanup`, the registry has no model `registry-test`, and the experiment `connectivity-test` is in the `deleted` stage.
- [ ] The registry never held a model named `churn` or an alias `champion` during this spec.
- [ ] `kubectl get pod,cm -n kubeflow` lists no `mlflow-check` or `mlflow-check-code`.
- [ ] After the cleanup, the server pod is `Running`, the PVC is `Bound` and the Service `mlflow` exists.

### Docs and hygiene
- [ ] The `DECISIONS.md` log has a dated line with the MLflow `3.17.2` pin and the image, and `git diff main -- DECISIONS.md` shows no change to the frozen contracts table.
- [ ] `docs/runbook.md` has a "MLflow" section with the commands run, the observed results and the gotchas.
- [ ] README §4 no longer says "to install" for the MLflow server and its PVC.
- [ ] `minikube image ls | grep churn-base` shows only `churn-base:0.1.0` (the image was not rebuilt).
- [ ] `git ls-files | grep -c '\.csv$'` returns 0.

## Decisions

- **Yes:** all of E1 (E1.1–E1.4) in one spec. Four small tasks on one component, with one "done when".
- **Yes:** official image `ghcr.io/mlflow/mlflow:v3.17.2`. No Docker build is needed, and the server needs the full `mlflow` package, which `churn-base` deliberately does not carry.
- **No:** a custom server image up front. It would need a build by the user and only pays off if the official image has no arm64 variant. The step 2 check settles that.
- **Yes:** server `3.17.2`, the same version as the `mlflow-skinny` client in `churn-base`. It avoids API mismatches between the two.
- **No:** `latest` or an unpinned tag. A silent upgrade could break the client.
- **Yes:** SQLite on the PVC as backend store. The README plans parallelism 2, which limits write contention.
- **No:** Postgres now. It is the README fallback, and it only matters if `database is locked` shows up in E2.
- **Yes:** `--serve-artifacts` with `--artifacts-destination /mlflow/artifacts`. Pods in `kubeflow` upload and download artifacts over HTTP and never mount `mlflow-data`.
- **No:** sharing the PVC with `kubeflow`, because a pod can only mount a PVC from its own namespace. **No** S3 or MinIO either: no credentials, and one more component to run.
- **Yes:** namespace `mlflow`, PVC of 2Gi (`ReadWriteOnce`, `standard`) and `strategy: Recreate`. A `RollingUpdate` would leave two pods fighting for a RWO volume.
- **Yes:** requests of 256Mi memory and 100m CPU, and a memory limit of 1Gi. The node has about 8 GB and KFP, KServe and later Prometheus share it. The numbers are an estimate, adjusted in step 2 if the server exceeds them.
- **Yes:** `--workers 1` and `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`, added in step 2. With the defaults, MLflow 3.17.2 starts several uvicorn workers and about 6 `huey` job consumers (about 160 MB each). That was over 1.5 GB: the container was `OOMKilled` at 1Gi, and at 2Gi it left the node thrashing. Nothing in this project needs the job executor. The memory limit goes back to 1Gi.
- **Yes:** `--allowed-hosts mlflow.mlflow.svc.cluster.local:5000,mlflow.mlflow.svc.cluster.local,localhost:*,127.0.0.1:*`, added in step 3. MLflow 3.17.2 refused the in-cluster host name with 403 "Invalid Host header" (the risk listed below). The `localhost` entries keep the port-forward working. **No** `*`.
- **Yes:** readiness probe on `/health`. `kubectl wait` then means the server really answers.
- **Yes:** the local UI goes through `port-forward` on port 5001. Port 8080 is held by another port-forward in this setup, and 5000 is commonly taken by the AirPlay Receiver on macOS.
- **Yes:** `tools/mlflow_check.py` shipped through a ConfigMap to one `sleep` pod, with `kubectl exec` per mode. It follows the SPEC 02 pattern, so the image is not rebuilt. The pod survives between steps, so the UI can be checked before the cleanup.
- **No:** `kubectl run ... python -c`, which is fragile for a script this long. **No** one pod per mode either.
- **Yes:** test names `connectivity-test`, `registry-test` and alias `check`.
- **No:** the names `churn` or `champion`. They are frozen contracts, and a test must not leave a fake champion in the registry.
- **Yes:** cleanup through the API only. MLflow keeps deleted runs and the experiment as soft-deleted rows, and the name `connectivity-test` cannot be reused. The user accepted that trace.
- **No:** `mlflow gc` in the server pod.
- **Yes:** the server, its PVC and its Service stay installed after the spec. E2 needs them.
- **Yes:** the `DECISIONS.md` log line comes before any manifest, as the repo rule asks.
- **No:** `MLFLOW_TRACKING_URI` in `components/config.py` now. E2.1 adds it, together with the experiment and model names.

## Risks

| Risk | Mitigation |
|---|---|
| The official image may have no arm64 variant, and the Mac is Apple Silicon. The pod would fail with `ImagePullBackOff` or `exec format error`. | Step 2 saves the pod events as evidence and stops. The custom-image fallback needs its own decision (and a `minikube image build` run by the user) before it is built. |
| MLflow 3.x validates the `Host` header. Calls to `mlflow.mlflow.svc.cluster.local:5000` may be refused (for example a 403 "Invalid Host header") even though the server is healthy. | Step 3 shows it as an error from the pod. If it happens, add the in-cluster host name to the server's allowed hosts, log it in `DECISIONS.md` and the runbook, and re-run step 2. |
| The `mlflow-skinny` client in `churn-base` may lack something the registry or `models:/` loading needs. | Step 4 shows it. Pin or add only what is missing, and log the change in `DECISIONS.md`. The full `mlflow` package does not go into `churn-base` without a decision, because the image tag would change. |
| The soft-deleted experiment `connectivity-test` blocks its own name, so the check cannot be re-run unchanged. | Accepted trace (see Decisions). A re-run uses a different experiment name in the script, and the change is noted in the runbook. |
| MLflow adds memory pressure on top of KFP and KServe, and Prometheus and Grafana come later. | Requests and limits are set (256Mi / 1Gi). The acceptance criteria check that no pod leaves `Running`. If the node is tight, apply the README memory fallback. |
| The PVC uses minikube's hostpath, so `minikube delete` loses every run and model. | Nothing in this spec is worth keeping, and E2 can recreate it. The runbook states it, as it did for `churn-data`. |
| Pods are immutable, so after editing `mlflow_check.py` the old code stays in the running pod. | Steps 4 and 5 recreate the ConfigMap and the pod before each new mode. |

## What is **not** in this spec

- The MLflow constants (tracking URI, experiment name, model name) in `components/config.py`, next to the rest of the shared configuration. They are fixed values, unlike run ids, which MLflow generates on every run. They land in E2.1.
- Pipeline 1, Optuna studies, and the real `churn` model with `@champion` (E2).
- A custom MLflow server image, unless the official one fails on arm64 and a new decision says otherwise.
- Postgres, S3 or MinIO, authentication, Ingress and backups.
- `mlflow gc` and permanent removal of the soft-deleted test rows.
- MLflow metrics in Prometheus (E4).

Each one of those, if it lands, goes in its own spec.
