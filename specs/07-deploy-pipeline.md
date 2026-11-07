# SPEC 07 — Deploy pipeline (serve the champion)

> **Status:** Approved
> **Depends on:** SPEC 01, SPEC 02, SPEC 03, SPEC 04, SPEC 05, SPEC 06
> **Backlog:** E3 (Pipeline 2 — Deploy), tasks E3.1 to E3.5 of `BACKLOG.md`, plus the derived serving image for XGBoost and LightGBM champions that SPEC 05 and SPEC 06 left to this spec
> **Date:** 2026-11-06
> **Objective:** Add a deploy pipeline that saves `models:/churn@champion` as `model.joblib` on the PVC, publishes it as the KServe InferenceService `churn` (with a derived serving image that also loads XGBoost and LightGBM), and smoke-tests that `:predict` returns the probabilities of that exact model.

## Why this spec exists

SPEC 06 left `churn@champion` in the registry. This spec takes it from there to an endpoint that answers `:predict`. It looks like a plain deploy, but three things are not obvious:

- **A champion of any family must be servable.** The KServe `sklearnserver:v0.21.0` image has neither `xgboost` nor `lightgbm`, so a registered `Pipeline` of those families fails to load there. SPEC 05 and SPEC 06 left the fix to this spec: a derived image with the same library versions as `churn-base:0.3.0`, used through a `ServingRuntime` of its own. The cluster's `kserve-sklearnserver` runtime is not touched.
- **`Ready` does not prove which version is served.** After a `patch`, the InferenceService can report `Ready` while the old revision still answers. So the smoke test compares the endpoint's probabilities with `predict_proba` of the file that Fetch just saved. If they match, the new version is the one being served.
- **The version in the folder name must be the version in the file.** Fetch reads the number from the alias and then loads that exact version, so a champion that changes while the pipeline runs cannot put version 3 inside the folder `v2`.

The pipeline has no parameters and always deploys `churn@champion`. Rollback, metrics, the load generator and automatic triggers are other specs.

## Scope

**In:**

- **E3.1 RBAC (`k8s/rbac-pipeline-runner.yaml`):** a `Role` and a `RoleBinding` in `kubeflow` that let the service account `pipeline-runner` get, create and patch `inferenceservices` (and get, list `servingruntimes`).
- **Derived serving image (`docker/churn-sklearnserver/Dockerfile`):** `kserve/sklearnserver:v0.21.0` plus `xgboost==3.2.0`, `lightgbm==4.7.0` and `libgomp1` if the image lacks it, tagged `churn-sklearnserver:0.1.0`. The versions match `churn-base:0.3.0`. The user builds it with `minikube image build`.
- **Serving runtime (`k8s/serving-runtime.yaml`):** a `ServingRuntime` `churn-sklearnserver` in `kubeflow` that copies the arguments of the cluster's `kserve-sklearnserver` runtime and uses the derived image.
- **E3.2 Fetch champion (`components/fetch_champion.py`):** loads `models:/churn@champion`, resolves its version number `N`, and writes `models/churn/v<N>/model.joblib` on the PVC with `joblib.dump`. That folder holds only that file.
  - Outputs: `model_version` (`str`), `model_dir` (`str`, the path on the PVC) and `storage_uri` (`str`, `pvc://churn-data/models/churn/v<N>`).
  - It checks that `predict_proba` of the reloaded file returns shape `(n, 2)`.
- **E3.3 Publish (`components/publish.py`):** creates or patches the InferenceService `churn` in `kubeflow` with `runtime: churn-sklearnserver`, the `storage_uri` of Fetch and `PREDICT_PROBA=true`, then waits until it is `Ready` and the rollout of the predictor Deployment is complete. It builds the manifest in code, because the `storageUri` changes with every version.
  - Output: `endpoint` (`str`, the in-cluster URL of `:predict`).
- **E3.4 Smoke test (`components/smoke_test.py`):** takes 3 rows of `train.csv` in `FEATURES` order, posts them to `:predict`, and compares the answer with `predict_proba` of the saved `model.joblib` on the same rows.
  - It fails if the answer is not 3 pairs of probabilities in [0, 1], or if the maximum difference with the local result is 1e-6 or more. That difference is what proves the endpoint serves the new version and not the previous one.
- **E3.5 Pipeline wiring (`pipelines/deploy_pipeline.py`):** `fetch_champion → publish → smoke_test`, with no parameters, every task through the same `_configure` helper as Pipeline 1. `tools/submit_pipeline.py` takes a `--pipeline` argument (`training` by default, or `deploy`) so one tool submits both.
- **Run and verify on the cluster:**
  - A first run creates the InferenceService, and a second run with the same champion patches it and ends `Succeeded`.
  - `curl` from the laptop through a port-forward returns probabilities.
  - The derived image is checked against a `model.joblib` of each of the five families, so an XGBoost or LightGBM champion is known to load before one wins.
- **Docs:** a "Deploy pipeline" section in `docs/runbook.md`, dated lines in the `DECISIONS.md` log (image tag, runtime), the README rows (InferenceService, derived image, and the XGBoost/LightGBM risk of §7 marked as solved), and the ticks of E3.1 to E3.5 in `BACKLOG.md`.

**Out of scope (for future specs):**

- A `model_version` parameter or any rollback from the pipeline. Going back is a manual edit of the `storageUri`.
- Deleting old `models/churn/v<N>/` folders. They stay on the PVC.
- Prometheus metrics, the scrape config, Grafana and the check that the predictor exposes `/metrics` (E4).
- The load generator and `submission.csv` (E4).
- Triggering Deploy automatically after Training, or any CI/CD.
- Canary or traffic splitting, Istio, ingress, authentication and autoscaling settings.
- A custom predictor. The stock `sklearnserver` with `PREDICT_PROBA=true` is enough.
- Any change to Pipeline 1, to Register, to the gate rule or to the frozen contracts.
- The full 30-trial run, still deferred to the end of the project.

## Data model

This spec adds no business data. It defines the contract of the three components, the new constants, the objects in the cluster and the layout of the PVC. The frozen contracts of `DECISIONS.md` do not change.

### Constants added to `components/config.py`

```python
PVC_CLAIM_NAME = "churn-data"                 # claim behind pvc:// URIs
K8S_NAMESPACE = "kubeflow"                    # where the InferenceService and the runtime live
ISVC_NAME = "churn"
SERVING_RUNTIME_NAME = "churn-sklearnserver"
PUBLISH_TIMEOUT_S = 300                       # how long Publish waits for Ready
SMOKE_ROWS = 3
SMOKE_TOLERANCE = 1e-6                        # max |endpoint - local predict_proba|
```

`MODELS_DIR_PVC`, `MODEL_NAME`, `CHAMPION_ALIAS`, `FEATURES` and `MLFLOW_TRACKING_URI` are reused unchanged. `config.py` already travels in the ConfigMap `churn-code`, so `tools/sync_code.sh` does not change.

### Components (contract)

| Component | Inputs | Outputs | Fails when |
|---|---|---|---|
| `fetch_champion` | none | `model_version` (`str`), `model_dir` (`str`), `storage_uri` (`str`) | MLflow is unreachable, the alias `champion` does not exist, the saved file does not reload, or `predict_proba` on a few rows has a shape other than `(n, 2)` |
| `publish` | `storage_uri` | `endpoint` (`str`) | the runtime `churn-sklearnserver` does not exist, the API call is forbidden, or the InferenceService is not `Ready` within `PUBLISH_TIMEOUT_S` |
| `smoke_test` | `endpoint`, `model_dir` | none | the HTTP status is not 200, the answer is not `SMOKE_ROWS` pairs of floats in [0, 1], or the maximum difference with the local `predict_proba` is `SMOKE_TOLERANCE` or more |

- `model_dir` is `/mnt/data/models/churn/v<N>`, `storage_uri` is `pvc://churn-data/models/churn/v<N>` and `endpoint` is `http://churn-predictor.kubeflow.svc.cluster.local/v1/models/churn:predict`.
- `<N>` is the MLflow version number behind `models:/churn@champion`, read with `MlflowClient.get_model_version_by_alias`. Fetch loads by version (`models:/churn/<N>`) once it knows `N`, so the file and the folder name cannot disagree if the alias moves in between.
- `model_dir` holds exactly one file, `model.joblib`, because `sklearnserver` takes one model file per folder. If the folder already exists, Fetch overwrites that file and nothing else. It writes `model.joblib.tmp` in the same folder and renames it with `os.replace`, so a pod that restarts during the write never reads a half-written file.
- Smoke test takes the first `SMOKE_ROWS` rows of `processed/train.csv` in `FEATURES` order, with numbers as floats and strings as strings, as in the spike. The churn probability is `predictions[i][1]`. It uses `urllib` from the standard library, so it adds no dependency.
- Each component lives in its own file under `components/` and is a `@dsl.component(base_image="churn-base:0.3.0")` function. Imports go inside the body.
- Every task goes through the same `_configure` helper as Pipeline 1 (code ConfigMap, PVC, no caching, `OMP_NUM_THREADS=1`). `deploy_pipeline.py` imports the helper from `pipelines/training_pipeline.py`, which stays untouched.
- Resources: `fetch_champion` asks 500m CPU and 512Mi with a 1.5Gi limit, like `register`. `publish` and `smoke_test` ask 100m and 256Mi with a 1Gi limit.

### Pipeline `churn-deploy`

```text
fetch_champion → publish(storage_uri) → smoke_test(endpoint, model_dir)
```

No parameters.

### Cluster objects

| Object | Namespace | Notes |
|---|---|---|
| Image `churn-sklearnserver:0.1.0` | — | `kserve/sklearnserver:v0.21.0` plus `xgboost==3.2.0` and `lightgbm==4.7.0` (and `libgomp1` only if the base lacks it), built by the user |
| `ServingRuntime` `churn-sklearnserver` | `kubeflow` | Arguments copied from the cluster's `kserve-sklearnserver`; only the name and the image change |
| `Role` and `RoleBinding` | `kubeflow` | `pipeline-runner` may `get`, `create` and `patch` `inferenceservices`, and `get` and `list` `servingruntimes` |
| `InferenceService` `churn` | `kubeflow` | Created and patched by `publish`, never by a manifest in the repo |
| Pipeline `churn-deploy` | `kubeflow` | Three steps |

InferenceService built by `publish` (shape, `<N>` filled in at run time):

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn
  namespace: kubeflow
  annotations:
    churn/model-version: "<N>"        # for humans: which version is served
spec:
  predictor:
    minReplicas: 1
    model:
      modelFormat: {name: sklearn}
      runtime: churn-sklearnserver
      storageUri: pvc://churn-data/models/churn/v<N>
      env:
        - {name: PREDICT_PROBA, value: "true"}
      resources:
        requests: {cpu: 100m, memory: 256Mi}
        limits: {memory: 1Gi}
```

### Layout on the PVC `churn-data`

```text
/mnt/data/models/churn/v<N>/model.joblib     # one folder per deployed version, never deleted here
```

The spike folder `models/spike/v1/` was part of SPEC 02 and stays out of this layout.

## Implementation plan

1. **Constants and decision lines.**
   - Add the seven constants of the data model to `components/config.py`.
   - Add two lines to the `DECISIONS.md` log: the image `churn-sklearnserver:0.1.0` (the `sklearnserver` v0.21.0 image plus `xgboost 3.2.0` and `lightgbm 4.7.0`, closing the open item of SPEC 05 and SPEC 06) and the `ServingRuntime` `churn-sklearnserver` that uses it.
   - Check: `uv run python -c "from components import config"` exits 0, and `FEATURES`, `SEED`, `MODEL_NAME` and `CHAMPION_ALIAS` are unchanged.
2. **Check the assumptions on the cluster (no repo changes).**
   - Run `minikube start`, wait for the pods to settle, and record in `docs/runbook.md`:
     - the service account that KFP step pods use (expected `pipeline-runner`);
     - that `churn-base:0.3.0` imports `kubernetes` (a throwaway pod, deleted afterwards);
     - the arguments of `clusterservingruntime kserve-sklearnserver`;
     - the user of `kserve/sklearnserver:v0.21.0`, whether `pip` works in it, and whether `libgomp.so.1` is present (a throwaway pod).
   - Check: every item has a recorded answer. If the service account is not `pipeline-runner`, or `kubernetes` is missing from `churn-base`, stop and bring the options to the user, because step 3 and the image of `churn-base` depend on it.
3. **RBAC (E3.1).**
   - Write `k8s/rbac-pipeline-runner.yaml` with the Role and RoleBinding of the data model, and apply it.
   - Check: `kubectl auth can-i create inferenceservices -n kubeflow --as=system:serviceaccount:kubeflow:pipeline-runner` prints `yes`, the same for `patch` and `get`, and `can-i delete inferenceservices` prints `no`.
4. **Derived serving image (the user builds it).**
   - Write `docker/churn-sklearnserver/Dockerfile` from what step 2 found, installing with `pip install --no-deps` from a pinned list, and add a short build command to `docs/runbook.md`.
   - The user runs `minikube image build` with the tag `churn-sklearnserver:0.1.0`.
   - Write `tools/serving_image_check.py`: fits the five families of `tuning_config.json` on 2,000 rows with `models.build_pipeline`, saves each as `models/image-check/<family>/model.joblib` on the PVC, and later loads each one and compares `predict_proba` with the expected values.
   - Check: `minikube image ls | grep churn-sklearn` shows the tag. A pod with `churn-base:0.3.0` writes the five files. A pod with `churn-sklearnserver:0.1.0` loads the five and the maximum difference is below `SMOKE_TOLERANCE` for every family. `pip freeze` in the derived image differs from the original only by the added packages: `scikit-learn` stays at `1.5.2` and `numpy` at the server's version. Then the same check with the original `kserve/sklearnserver:v0.21.0` fails for `xgboost` and `lightgbm` with `ModuleNotFoundError`, which proves the derived image is what fixes it. Delete the pods.
5. **Serving runtime.**
   - Write `k8s/serving-runtime.yaml` from the arguments recorded in step 2, with the new name and image, and apply it.
   - Check: a temporary InferenceService `churn-image-check` that uses the runtime and `pvc://churn-data/models/image-check/xgboost` becomes `Ready`, and `:predict` on 3 rows returns shape `(3, 2)` equal to the local result. Delete it and the `models/image-check/` folder afterwards.
6. **Fetch champion component (E3.2).**
   - Write `components/fetch_champion.py`: read the version behind the alias, load `models:/churn/<N>`, write `model.joblib.tmp` to `models/churn/v<N>/` and rename it to `model.joblib` with `os.replace`, reload the final file, check the shape of `predict_proba`, and return the three outputs.
   - Check, locally, against a temporary SQLite MLflow store and a temporary models folder (the constant pointed there):
     - With one registered version, the outputs name version `1`, the folder holds only `model.joblib`, and `storage_uri` ends in `models/churn/v1`.
     - With two versions and the alias on `2`, the folder is `v2` and `v1` is not touched.
     - Running it twice for the same version overwrites the file and leaves nothing else in the folder (no `.tmp` left).
     - With no alias `champion`, it raises.
     - It works with a `logreg` and an `xgboost` champion.
7. **Publish component (E3.3).**
   - Write `components/publish.py`: check the runtime exists, build the manifest of the data model, create the InferenceService if it does not exist or patch its `storageUri` and annotation if it does, then wait, up to `PUBLISH_TIMEOUT_S`, until `Ready` is true, `observedGeneration` matches the generation **and** the Deployment `churn-predictor` has finished its rollout (its `observedGeneration` is current and its `replicas`, `updatedReplicas` and `availableReplicas` all equal the wanted replicas). Return `endpoint`.
   - Check, locally, with the Kubernetes client replaced by a stub: a missing InferenceService produces a `create` with the exact shape of the data model, an existing one produces a `patch` with the new `storageUri`, a missing runtime raises, and a `Ready` that never arrives raises after the timeout.
8. **Smoke test component (E3.4).**
   - Write `components/smoke_test.py`: read 3 rows of `train.csv` in `FEATURES` order, post them with `urllib`, load `model_dir/model.joblib`, compare.
   - Check, locally, against a throwaway HTTP server that returns canned answers: a correct answer passes. A status other than 200, a wrong shape, a probability above 1 and a difference of `SMOKE_TOLERANCE` or more each raise.
9. **Pipeline wiring (E3.5, first part).**
   - Write `pipelines/deploy_pipeline.py` as in the data model, importing `_configure` from `pipelines/training_pipeline.py`. Add `--pipeline training|deploy` to `tools/submit_pipeline.py`, with `training` as the default.
   - Check: both pipelines compile (`--compile-only`). The deploy YAML has the three tasks in the order `fetch_champion → publish → smoke_test`, references `churn-base:0.3.0` and not `0.2.0`, and has no pipeline parameters. `git diff main -- pipelines/training_pipeline.py` is empty.
10. **Runs on the cluster (E3.5, second part).**
    - Open `kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888`, run `tools/sync_code.sh`, and run `tools/submit_pipeline.py --pipeline deploy --wait`.
    - Check: the run is `Succeeded`, `isvc/churn` is `Ready`, its annotation names the version behind `churn@champion`, and `models/churn/v<N>/model.joblib` exists on the PVC.
    - Run it a second time with the same champion. Check: `Succeeded`, the InferenceService was patched and not recreated (same `uid`), and it is `Ready`.
    - Forward the predictor to a free local port (not 8080, which the KFP UI may hold) and `curl` it with one raw row: the answer is `{"predictions": [[p0, p1]]}` with both values in [0, 1]. Note the time of both runs.
    - If a step fails on RBAC, the image, memory or a `Ready` that flashes true before the new revision, save the pod logs and events as evidence and stop.
11. **Docs and hygiene.**
    - Write the "Deploy pipeline" section of `docs/runbook.md` with the commands actually run, the answers of step 2, the observed results, the times and the gotchas. Update the README rows (InferenceService, `churn-sklearnserver` image, §7 XGBoost and LightGBM risk marked as solved) and tick E3.1 to E3.5 in `BACKLOG.md`.
    - Check: every command in the runbook section appears in the steps above, no temporary pod or `churn-image-check` object is left in `kubeflow`, and `models/image-check/` is gone from the PVC.

## Acceptance criteria

### Config and decisions
- [ ] `components/config.py` defines `PVC_CLAIM_NAME`, `K8S_NAMESPACE`, `ISVC_NAME`, `SERVING_RUNTIME_NAME`, `PUBLISH_TIMEOUT_S`, `SMOKE_ROWS` and `SMOKE_TOLERANCE`, and `FEATURES`, `SEED`, `MODEL_NAME` and `CHAMPION_ALIAS` are unchanged.
- [ ] The `DECISIONS.md` log has dated lines for the image `churn-sklearnserver:0.1.0` and for the `ServingRuntime` `churn-sklearnserver`, and `git diff main -- DECISIONS.md` shows no change to the frozen contracts table.

### Assumptions checked on the cluster
- [ ] `docs/runbook.md` records the service account of the KFP step pods, whether `churn-base:0.3.0` imports `kubernetes`, the arguments of `kserve-sklearnserver`, and the user, `pip` and `libgomp.so.1` of `kserve/sklearnserver:v0.21.0`.

### RBAC
- [ ] `kubectl auth can-i` as `system:serviceaccount:kubeflow:pipeline-runner` prints `yes` for `get`, `create` and `patch` on `inferenceservices` in `kubeflow`, and `no` for `delete`.

### Derived image and runtime
- [ ] `minikube image ls | grep churn-sklearn` shows `churn-sklearnserver:0.1.0`.
- [ ] `pip freeze` in the derived image differs from the original `kserve/sklearnserver:v0.21.0` only by the added packages, with `scikit-learn` still `1.5.2`.
- [ ] A pod with `churn-sklearnserver:0.1.0` loads a `model.joblib` of each of the five families (`logreg`, `random_forest`, `hist_gradient_boosting`, `xgboost`, `lightgbm`), and the maximum difference of `predict_proba` with the expected values is below `SMOKE_TOLERANCE` for every one.
- [ ] The same check with `kserve/sklearnserver:v0.21.0` fails for `xgboost` and `lightgbm` with `ModuleNotFoundError` and passes for the other three.
- [ ] `kubectl get servingruntime churn-sklearnserver -n kubeflow` exists and uses the image `churn-sklearnserver:0.1.0`.
- [ ] A temporary InferenceService with that runtime serving an `xgboost` `model.joblib` becomes `Ready`, and `:predict` on 3 rows returns shape `(3, 2)` equal to the local result.

### Fetch champion (local, temporary MLflow store)
- [ ] With one registered version, the outputs name version `1`, `model_dir` ends in `models/churn/v1` and holds only `model.joblib`, and `storage_uri` is `pvc://churn-data/models/churn/v1`.
- [ ] With two versions and the alias on `2`, Fetch writes `v2` and does not touch `v1`.
- [ ] Running Fetch twice for the same version overwrites the file and leaves nothing else in the folder (no `model.joblib.tmp`).
- [ ] With no alias `champion`, Fetch raises.
- [ ] Fetch saves and reloads a `logreg` champion and an `xgboost` champion, and `predict_proba` has shape `(n, 2)` for both.

### Publish (local, Kubernetes client stubbed)
- [ ] With no InferenceService, Publish calls `create` with a manifest that has `runtime: churn-sklearnserver`, the `storageUri` it received, `PREDICT_PROBA=true`, the annotation `churn/model-version` and the resources of the data model.
- [ ] With an existing InferenceService, Publish calls `patch` with the new `storageUri` and annotation, and does not call `create`.
- [ ] With a missing runtime, Publish raises before creating anything.
- [ ] With an InferenceService that never becomes `Ready`, Publish raises after `PUBLISH_TIMEOUT_S`.
- [ ] With an InferenceService that is `Ready` and has seen its generation but whose Deployment is still rolling out (replicas not all updated and available, or the Deployment `observedGeneration` behind), Publish keeps waiting, and it raises after `PUBLISH_TIMEOUT_S` with the Deployment numbers in the message if the rollout never finishes. A missing Deployment also raises.
- [ ] On the cluster, during a real rollout, Publish returns only after the new pod is Ready and the old pod no longer receives traffic (checked with a watcher on two rollouts).

### Smoke test (local, throwaway HTTP server)
- [ ] A correct answer (3 pairs in [0, 1], difference below `SMOKE_TOLERANCE`) passes.
- [ ] A status other than 200, a wrong shape, a probability above 1, and a difference of `SMOKE_TOLERANCE` or more each raise.
- [ ] `grep -c "import requests\|import httpx" components/smoke_test.py` prints 0 (standard library only).

### Pipeline on the cluster
- [ ] `tools/submit_pipeline.py --pipeline deploy --compile-only` and `--pipeline training --compile-only` both compile.
- [ ] The deploy YAML has the three tasks in the order `fetch_champion`, `publish`, `smoke_test`, references `churn-base:0.3.0` and not `0.2.0`, and has no pipeline parameters.
- [ ] `git diff main -- pipelines/training_pipeline.py` is empty.
- [ ] The first deploy run finishes `Succeeded`, `isvc/churn` is `Ready`, its annotation `churn/model-version` equals the version behind `churn@champion`, and `models/churn/v<N>/model.joblib` exists on the PVC.
- [ ] The second deploy run finishes `Succeeded`, and `isvc/churn` keeps the same `uid` as after the first run (patched, not recreated).
- [ ] `curl` on the predictor through a port-forward with one raw row in `FEATURES` order returns `{"predictions": [[p0, p1]]}` with both values in [0, 1].
- [ ] `kubectl get pods -A --no-headers | awk '$4!="Running" && $4!="Completed"'` prints nothing, no temporary pod or `churn-image-check` object is left in `kubeflow`, and `models/image-check/` is not on the PVC.

### Docs and hygiene
- [ ] `docs/runbook.md` has a "Deploy pipeline" section with the commands run, the answers of the cluster checks, the observed results, the times of both runs and the gotchas.
- [ ] The README rows of the InferenceService and the derived image are updated, and §7 says the XGBoost and LightGBM serving risk is solved.
- [ ] `BACKLOG.md` has E3.1 to E3.5 ticked.
- [ ] `git diff main -- components/tuning_config.json` is empty, and `git ls-files | grep -c '\.csv$'` returns 0.

## Decisions

- **Yes:** one spec for RBAC, the derived image, the runtime, three components and the wiring. They are one chain (champion → file → endpoint → check), and a half of it would leave nothing to verify.
- **No:** two specs, with the derived image apart. The pipeline would break the day an XGBoost or LightGBM champion wins, and SPEC 05 and SPEC 06 already promised the image to this spec.
- **Yes:** build the derived image `churn-sklearnserver:0.1.0` now, with the same `xgboost` and `lightgbm` versions as `churn-base:0.3.0`. Training and serving then load the same pickle with the same libraries, whichever family wins.
- **No:** waiting until a boosting model is champion. Today the champion is `hist_gradient_boosting`, so the gap would stay hidden until it fails in a deploy run.
- **Yes:** a `ServingRuntime` `churn-sklearnserver` in `kubeflow`, with the arguments of the cluster's `kserve-sklearnserver` and only the name and the image changed. The InferenceService names it with `runtime:`, as in the spike. It is declarative, and the image is in one place.
- **No:** overriding `image` inside the InferenceService. It is shorter, but it skips the runtimes mechanism of KServe and hides the image in a field that the next reader will not look at.
- **No:** editing the cluster's `ClusterServingRuntime` `kserve-sklearnserver`. It belongs to the KServe install and other InferenceServices could depend on it.
- **Yes:** the smoke test compares the endpoint's probabilities with `predict_proba` of the saved `model.joblib` on the same rows, with a tolerance of `SMOKE_TOLERANCE`. That proves the endpoint serves the new version and not the previous one, which `Ready` alone cannot show.
- **No:** a check that only looks at shape and range. A stale endpoint would pass it.
- **Yes:** the 3 rows come from `processed/train.csv` in `FEATURES` order. They are real rows with the right types, and the file is already on the PVC.
- **No:** fixed rows written in the code. They would be one more copy of the feature order to keep in step.
- **Yes:** the pipeline has no parameters and always deploys `churn@champion`, as README §4 says.
- **No:** a `model_version` parameter for rollback. It is useful, but it widens the spec, and the smoke test would have to work with old versions. Going back is a manual edit of the `storageUri`.
- **Yes:** old `models/churn/v<N>/` folders stay on the PVC. Each model is small, and they leave the manual rollback possible.
- **No:** deleting old folders in Deploy. A cleanup that runs by itself can remove the version someone wanted to go back to.
- **Yes:** Fetch reads the version number from the alias and then loads `models:/churn/<N>`. If the alias moves while the pipeline runs, the folder name and the file still agree. The contract (alias `champion`) does not change.
- **No:** loading `models:/churn@champion` and reading the version in a second call. A race between the two calls could put version 3 inside the folder `v2`.
- **Yes:** Fetch overwrites `model.joblib` if the folder exists and writes nothing else there once it finishes. `sklearnserver` takes exactly one model file per folder, as the spike found.
- **Yes:** Fetch writes `model.joblib.tmp` and renames it with `os.replace`. A predictor pod that restarts during the second run never reads a half-written file, and `sklearnserver` ignores the `.tmp` extension.
- **Yes:** Fetch reloads the saved file and checks `predict_proba` before returning. A bad file then fails here, close to its cause, and not inside the pod of the predictor.
- **Yes:** Publish builds the InferenceService manifest in code, with the `storageUri` of the version. It changes every time, so a YAML in the repo would be a template with a hole.
- **No:** a `k8s/inferenceservice.yaml` with the version edited by hand. A step that is manual is a step that gets forgotten.
- **Yes:** Publish creates or patches with the Python Kubernetes client, using the service account of the step pod. The RBAC of E3.1 gives it exactly that permission and no more (`get`, `create`, `patch`, no `delete`).
- **No:** `kubectl` inside the step. It would need a new image with a binary, for something the client does.
- **Yes:** Publish waits for `Ready`, for `observedGeneration` to match the generation **and** for the rollout of the predictor Deployment to be complete, up to `PUBLISH_TIMEOUT_S`. Measured on this cluster (`docs/runbook.md`, step 10): after a `patch` the InferenceService stays `Ready` from the old revision (16.7 s with a stale `observedGeneration` in one rollout), and in another rollout `observedGeneration` caught up while the new pod was not Ready and the old one was the only one answering. With `Ready` and `observedGeneration` alone, Publish returned about 5 s after the patch (5.5 s and 5.3 s in two rollouts), when the new pod needed 13 to 16 s to be Ready. With the Deployment condition it returned 15.7 s and 15.9 s after the patch, after the new pod was Ready and the old one stopped receiving traffic.
- **No:** `status.modelStatus.transitionStatus` as the rollout signal. It stayed `UpToDate` during the whole rollout in both measured rollouts, so it would have changed nothing.
- **No:** the smoke test retrying for a few seconds before it fails. It is more tolerant, but it can hide a slow or half-failed rollout. If `Ready` flashes true early on the cluster, step 10 stops and brings the options to the user.
- **Yes:** the smoke test uses `urllib` from the standard library. It adds no dependency to `churn-base`, and the call is one `POST`.
- **Yes:** the InferenceService carries the annotation `churn/model-version`. A person reading `kubectl get isvc churn -o yaml` sees which version is served without reading the `storageUri`.
- **Yes:** the Role also allows `get` and `list` on `servingruntimes`, so Publish can check that the runtime exists before it creates anything. A missing runtime then fails in seconds with a clear message.
- **No:** a wider Role (`*` or `delete`). The pipeline needs less, and the user can add more in a later spec.
- **Yes:** `deploy_pipeline.py` imports `_configure` from `pipelines/training_pipeline.py`. Both pipelines then configure their tasks the same way, and Pipeline 1 does not change.
- **No:** moving `_configure` to a shared `pipelines/common.py` now. It is a refactor of a file that is `Implemented` and not needed for this spec.
- **Yes:** `tools/submit_pipeline.py` takes `--pipeline training|deploy`, with `training` as the default, so one tool submits both and the existing command keeps working.
- **No:** a second script `submit_deploy.py`. It would copy the compile, sync and wait logic.
- **Yes:** the derived image is checked in three ways: a pod loads each of the five families, the original KServe image fails for `xgboost` and `lightgbm`, and a temporary InferenceService serves an `xgboost` file. The failure of the original image proves the derived one is what fixes it, and the InferenceService proves the runtime wiring.
- **No:** checking only that the libraries import. An import does not prove that a pickled `Pipeline` loads or that the runtime starts.
- **Yes:** the derived image installs the two libraries with `pip install --no-deps` from a pinned list, and step 4 compares `pip freeze` before and after. It is the lesson of SPEC 05, where a plain `pip install` pulled an unwanted dependency.
- **Yes:** the test files live in `models/image-check/` on the PVC and are deleted at the end, so the serving folders stay clean.
- **Yes:** the predictor asks 100m CPU and 256Mi with a 1Gi limit and runs one replica. The node is tight, and a bigger model is a later decision.
- **No:** autoscaling, Istio, ingress and authentication. They are outside the project scope in README §3.
- **Yes:** a custom predictor stays out. The stock `sklearnserver` with `PREDICT_PROBA=true` was verified in SPEC 02.
- **Yes:** the `curl` check of step 10 uses a free local port, not 8080. The KFP UI may hold 8080, and a `POST` there returns an HTML `404`, as the spike found.
- **Yes:** the assumptions that could not be read with the cluster stopped (service account name, `kubernetes` in `churn-base`, the runtime arguments, the user of the `sklearnserver` image) are checked in step 2, before anything depends on them. If one is wrong, the plan stops.
- **No:** a change to the frozen contracts. The registry names, `FEATURES` order and scikit-learn version are used as written, and the only new lines of `DECISIONS.md` are the image and the runtime.

## Risks

| Risk | Mitigation |
|---|---|
| After a `patch`, the InferenceService shows `Ready` from the old revision, Publish returns early and the smoke test hits the previous version. In the second run the `patch` changes nothing (same `storageUri`), so there is no rollout to wait for. | Publish also waits for `observedGeneration` to match the generation and for the predictor Deployment rollout to be complete (measured necessary in step 10: the first two checks alone returned while the old pod was the only one answering). The smoke test fails if the answer differs from the saved file by `SMOKE_TOLERANCE` or more, so a stale endpoint is caught and not accepted. Two consecutive model versions can give identical probabilities (versions 1 and 2 do: same seed, params and data), and then the comparison cannot tell them apart; the Deployment condition is what covers that case. |
| `kserve/sklearnserver:v0.21.0` runs as a non-root user, has no `pip` or cannot install `libgomp1`, so the derived image cannot be built the way step 4 expects. | Step 2 records the user, `pip` and `libgomp.so.1` before the Dockerfile is written. If the Dockerfile cannot be written with a `USER root` switch, stop and bring the options to the user. |
| `pip install xgboost lightgbm` moves `scikit-learn`, `numpy` or `scipy` inside the server image, and a pickle that loaded before no longer does. The same happened in SPEC 05 with an extra dependency (`nvidia-nccl-cu12`). | The Dockerfile installs with `pip install --no-deps` from a pinned list, and step 4 compares `pip freeze` before and after: `scikit-learn` stays at `1.5.2` and `numpy` at the server's version. Any other difference is a real change to explain. |
| A pickle written in `churn-base` (Python 3.11.18, numpy 2.4.6) does not load in the server (Python 3.11.16, numpy 2.4.4). The spike proved it for scikit-learn, but not for the XGBoost and LightGBM wrappers. | Step 4 loads a file of each of the five families inside the derived image and compares `predict_proba`. A failure there is found before any pipeline exists. |
| Fetch overwrites `model.joblib` of a version whose predictor pod is running (second run, same champion). If the pod restarts in the middle of the write, it reads a half-written file. | Fetch writes `model.joblib.tmp` in the same folder and renames it with `os.replace`. `sklearnserver` ignores the `.tmp` extension, and step 6 checks that nothing but `model.joblib` remains after the call. |
| The predictor does not fit in the node (1Gi limit): a Random Forest of 400 trees can weigh tens of MB, and the node is already shared with KFP, KServe and MLflow. The pod is `OOMKilled` and Publish times out. | Publish fails after `PUBLISH_TIMEOUT_S` with the InferenceService status in the message. The criteria check that no pod leaves `Running`. If it is killed, stop and bring the pod events and the model size to the user. Do not raise the limit inside this spec without a decision. |
| The service name or path is not what the data model says (`churn-predictor` on port 80, `/v1/models/churn:predict`), so the smoke test gets a connection error. | The spike showed `churn-spike-predictor` on port 80, the same pattern. Step 10 shows the error in the log of the step. If the name differs, the fix is the `endpoint` that Publish returns, and the spec is updated before the code. |
| The step pods use a service account other than `pipeline-runner`, so Publish gets `403` even with the Role applied. | Step 2 reads the real service account before step 3 writes the RoleBinding. The `can-i` check of step 3 uses that account. |
| KFP runs fail in their first minutes (drivers cannot reach `metadata-grpc` or MySQL). The cause was never found, and after a `minikube stop` the port-forwards die. | Resubmit. A Python traceback in a pod is a real failure and not this flake. Reopen the port-forwards, check `cpu.max`, and delete the Error pods that failed attempts leave. |
| The derived image build fails or is slow on Apple Silicon (arm64 wheels for `xgboost` or `lightgbm`). | `churn-base:0.3.0` already builds both for `aarch64`, so wheels exist. The user runs the build, and the runbook records the command and the time. |

## What is **not** in this spec

- A `model_version` parameter or any rollback from the pipeline.
- Deleting old `models/churn/v<N>/` folders.
- Prometheus metrics, the scrape config, Grafana and the check of `/metrics` (E4).
- The load generator and `submission.csv` (E4).
- Triggering Deploy automatically after Training, and any CI/CD.
- Canary, Istio, ingress, authentication and autoscaling.
- A custom predictor.
- Any change to Pipeline 1, to Register, to the gate rule or to the frozen contracts.
- The full 30-trial run, still deferred to the end of the project.

Each one of those, if it lands, goes in its own spec.
