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
