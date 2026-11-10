# SPEC 08 — Serving metrics into Prometheus

> **Status:** Approved
> **Depends on:** SPEC 01, SPEC 02, SPEC 03, SPEC 04, SPEC 05, SPEC 06, SPEC 07
> **Backlog:** E4 (Monitoring and load generator), tasks E4.1 to E4.3 of `BACKLOG.md`. E4.4 to E4.6 (load generator, Grafana dashboard, submission check) go in SPEC 09.
> **Date:** 2026-11-07
> **Objective:** Install a minimal kube-prometheus-stack with Helm in the namespace `monitoring` and scrape the `/metrics` endpoint of the `churn` predictor, so that its target shows UP in Prometheus.

## Why this spec exists

SPEC 07 left the `churn` predictor answering `:predict`. This spec makes its behaviour visible to Prometheus. It looks like a plain Helm install, but four things are not obvious:

- **The predictor already exposes metrics (E4.1, measured on 2026-11-07).** `GET /metrics` on port 8080 of the pod answers 200 with the histograms `request_preprocess_seconds`, `request_predict_seconds`, `request_postprocess_seconds` and `request_explain_seconds`, labelled `model_name="churn"`. No custom predictor is needed, and the fallback of README §7 is closed.
- **There is no error metric.** Those histograms count requests and time them, but no metric carries an HTTP status. The error-rate panel that the README and `BACKLOG.md` ask for cannot be built from them. This spec only records the gap; SPEC 09 decides how to cover it (for example counting errors in the load generator).
- **The node is tight.** On 2026-11-07 it had 2.0 GB of available memory (5.8 GB used of 7.8 GB) and 5.2 GB of free disk (91 % used). The stack is therefore trimmed to Prometheus, its operator and Grafana, and the spec measures memory and disk before and after the install. If it does not fit, it stops and brings the options to the user, because raising the node memory is a decision of its own.
- **The scrape goes through the Service, not the pod.** The container declares no named port, and the annotations `prometheus.kserve.io/*` of the pod are not read by the operator (`enable-prometheus-scraping` is `"false"`). The Service `churn-predictor` does have a named port (80 → 8080), so a `ServiceMonitor` on it works with no change to Publish or to the InferenceService.

This spec ends when Prometheus shows the target UP and a query returns the request count of the model. The dashboard, the load generator and the submission are SPEC 09.

## Scope

**In:**

- **E4.1 Check serving metrics (already measured):** the answer of `/metrics`, the metric families found and the missing error metric are written down in `docs/runbook.md`. No code is needed.
- **E4.2 Helm and kube-prometheus-stack:**
  - Helm 4.3.1 is installed (`brew install helm`, done by the user).
  - `k8s/monitoring-values.yaml` installs `kube-prometheus-stack` in the namespace `monitoring`, with the chart version pinned to one exact version.
  - Alertmanager, node-exporter and kube-state-metrics are off.
  - Prometheus keeps 24 h of data on a PVC of 2Gi (StorageClass `standard`), with a memory limit.
  - Grafana is **not** installed in the cluster (`grafana.enabled: false`). It runs in a Docker container on the user's Mac (Docker Compose, `docker/grafana/compose.yaml`) and reads Prometheus through the port-forward `19090`. See the decision below: Grafana 13 starved the node in the cluster.
  - Prometheus is set to pick up `ServiceMonitor` objects that do not carry the release label.
  - Memory (`minikube ssh -- free -m`) and disk (`df -h /var`) are measured before and after the install and written in the runbook.
- **E4.3 Scrape config:**
  - `k8s/servicemonitor-churn.yaml`, a `ServiceMonitor` for the Service `churn-predictor`, applied by hand.
  - `k8s/networkpolicy-allow-monitoring.yaml`, a `NetworkPolicy` in `kubeflow` that lets the namespace `monitoring` reach the predictor on port 8080, applied by hand. Without it the scrape times out (the namespace has `default-allow-same-namespace`, which only accepts ingress from the same namespace).
  - The target shows UP in Prometheus.
  - A PromQL query on `request_predict_seconds_count{model_name="churn"}` returns a value, and one `:predict` request through a port-forward raises it by 1.
  - Grafana on the Mac (`http://localhost:3000`) has a datasource of type `prometheus` pointing at `http://localhost:19090` whose health check answers `OK`.
- **Docs:**
  - A "Monitoring" section in `docs/runbook.md` with the commands actually run, the E4.1 findings, the memory and disk numbers and the gotchas of Helm 4.
  - A dated line in the `DECISIONS.md` log for the chart version.
  - README rows for Prometheus and Grafana marked as installed, and the risk "Metrics from KServe sklearn server" in §7 marked as solved.
  - E4.1 to E4.3 ticked in `BACKLOG.md`.

**Out of scope (for future specs):**

- The Grafana dashboard, its JSON, the screenshot and the choice of how to cover the error rate (SPEC 09, E4.5).
- The load generator, `submission.csv` and its check (SPEC 09, E4.4 and E4.6).
- A custom predictor. The stock `sklearnserver` already exposes `/metrics`.
- Alertmanager, alert rules, node-exporter and kube-state-metrics.
- Scraping anything other than the `churn` predictor (KFP, MLflow, the KServe controller).
- Long retention, remote storage, high availability, ingress and authentication for Prometheus or Grafana.
- Any change to Publish, to the InferenceService, to Pipeline 1 or to Pipeline 2.
- Raising the node memory or resizing the Docker Desktop disk. If the stack does not fit, that is a decision brought to the user.
- Cleaning Docker images or build cache. If the disk check needs it, the user runs it.
- The full 30-trial run, still deferred to the end of the project.

## Data model

This spec adds no business data and no Python code. It defines the files, the values of the chart, the `ServiceMonitor`, the objects in the cluster and the metrics that Prometheus will hold. The frozen contracts of `DECISIONS.md` do not change.

### Files added

| File | Purpose |
|---|---|
| `k8s/monitoring-values.yaml` | Values for `kube-prometheus-stack` (below) |
| `k8s/servicemonitor-churn.yaml` | The `ServiceMonitor` for the predictor (below) |
| `k8s/networkpolicy-allow-monitoring.yaml` | The `NetworkPolicy` that lets Prometheus scrape the predictor (below) |
| `docker/grafana/compose.yaml` | Grafana 13.3.0 in a container on the Mac, port 3000, named volume `grafana-data` (below) |

`docs/runbook.md`, `DECISIONS.md`, `README.md` and `BACKLOG.md` change as listed in the scope.

### Helm release

| Item | Value |
|---|---|
| Repository | `prometheus-community` → `https://prometheus-community.github.io/helm-charts` |
| Chart | `kube-prometheus-stack`, version **92.22.0** (app v0.94.2), pinned in the command and in the runbook |
| Release name | `monitoring` |
| Namespace | `monitoring` (created by `--create-namespace`) |
| Helm | 4.3.1 |

### `k8s/monitoring-values.yaml` (shape; the memory numbers are the starting point and are confirmed by the measurements of steps 2 and 6)

```yaml
defaultRules: {create: false}
alertmanager: {enabled: false}
nodeExporter: {enabled: false}
kubeStateMetrics: {enabled: false}
kubeApiServer: {enabled: false}
kubelet: {enabled: false}
kubeControllerManager: {enabled: false}
kubeScheduler: {enabled: false}
kubeProxy: {enabled: false}
kubeEtcd: {enabled: false}
coreDns: {enabled: false}
kubeDns: {enabled: false}

prometheusOperator:
  resources:
    requests: {cpu: 50m, memory: 64Mi}
    limits: {memory: 192Mi}

prometheus:
  prometheusSpec:
    retention: 24h
    serviceMonitorSelectorNilUsesHelmValues: false   # pick up ServiceMonitors without the release label
    podMonitorSelectorNilUsesHelmValues: false
    resources:
      requests: {cpu: 100m, memory: 256Mi}
      limits: {memory: 640Mi}
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: standard
          accessModes: [ReadWriteOnce]
          resources: {requests: {storage: 2Gi}}

grafana: {enabled: false}   # runs outside the cluster (container on the Mac), see Decisions
```

- The default scrape targets of the cluster (API server, kubelet, control plane, CoreDNS) and the default rules are off. They would add many series and memory, and nothing in this project uses them. Prometheus still scrapes itself, which is enough to prove the stack works before the predictor target exists.
- Grafana is not part of the release, so there is no Grafana password in the file, in a Secret or in the repo. The local Grafana asks for a new password at its first login.
- The datasource of the local Grafana is created by hand (URL `http://host.docker.internal:19090`), because the chart no longer provisions it.

### `docker/grafana/compose.yaml`

```yaml
services:
  grafana:
    image: grafana/grafana:13.3.0     # same version the user had with Homebrew
    ports: ["3000:3000"]
    volumes: [grafana-data:/var/lib/grafana]
volumes:
  grafana-data:
```

- The container reaches Prometheus through `host.docker.internal:19090`, where the port-forward listens. The port-forward must be open whenever Grafana is used.
- Docker commands are run by the user. The first login asks for a new admin password, and it is not stored in the repo.

### `k8s/servicemonitor-churn.yaml`

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: churn-predictor
  namespace: monitoring
spec:
  namespaceSelector:
    matchNames: [kubeflow]
  selector:
    matchLabels:
      serving.kserve.io/inferenceservice: churn
      component: predictor
  endpoints:
    - port: churn-predictor      # the named port of the Service (80 → 8080)
      path: /metrics
      interval: 15s
```

- The labels are those of the Service `churn-predictor` read on 2026-11-07 (`app=isvc.churn-predictor`, `component=predictor`, `serving.kserve.io/inferenceservice=churn`).
- The object lives in `monitoring` and points at `kubeflow` with `namespaceSelector`, so everything that belongs to monitoring is removed together.
- It is applied by hand and no component of Pipeline 2 creates it.

### `k8s/networkpolicy-allow-monitoring.yaml`

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-monitoring-scrape
  namespace: kubeflow
spec:
  podSelector:
    matchLabels:
      serving.kserve.io/inferenceservice: churn
      component: predictor
  policyTypes: [Ingress]
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
      ports:
        - {protocol: TCP, port: 8080}
```

- Found on 2026-11-09 in step 4: the target was `down` with `context deadline exceeded` while `/metrics` answered in 38 ms through a port-forward. The cause is the pre-existing `NetworkPolicy` `default-allow-same-namespace` in `kubeflow` (`podSelector: {}`, ingress only from `podSelector: {}`), enforced by kindnet.
- NetworkPolicies add up, so this object only opens port 8080 of the predictor pods to the namespace `monitoring`. The existing policy is not touched, and neither are the Service or the InferenceService.

### Cluster objects

| Object | Namespace | Notes |
|---|---|---|
| Helm release `monitoring` | `monitoring` | Prometheus operator and Prometheus (1 pod). No Grafana |
| CRDs `monitoring.coreos.com` | cluster | Installed by the chart; they stay if the release is uninstalled |
| PVC of Prometheus | `monitoring` | 2Gi, StorageClass `standard` (hostPath on the same node disk) |
| `ServiceMonitor` `churn-predictor` | `monitoring` | Scrapes the Service `churn-predictor` in `kubeflow` |

### What Prometheus will hold for the predictor

| Metric | Type | Labels | Used for (SPEC 09) |
|---|---|---|---|
| `request_predict_seconds` | histogram (`_bucket`, `_count`, `_sum`) | `model_name="churn"` | request rate and p50/p95 latency |
| `request_preprocess_seconds`, `request_postprocess_seconds`, `request_explain_seconds` | histogram | `model_name="churn"` | not used |
| `process_*`, `python_*` | counters and gauges | — | not used |

Prometheus adds `job="churn-predictor"`, `namespace="kubeflow"`, `service`, `pod` and `endpoint`. No metric carries an HTTP status.

### Local access (port-forwards, never 8080)

| Service | Command target | Local port |
|---|---|---|
| Prometheus | `svc/monitoring-kube-prometheus-prometheus` 9090 | `19090` |
| Grafana (outside the cluster) | `docker compose up -d`, `http://localhost:3000` | `3000` |

### Measurements recorded in the runbook

| Measure | Command | Before | After |
|---|---|---|---|
| Node memory | `minikube ssh -- free -m` | 5833 used / 2000 available (2026-11-07) | measured in steps 2 and 6 |
| Node disk | `minikube ssh -- df -h /var` | 5.2 GB free (91 %) | measured in steps 2 and 6 |

## Implementation plan

1. **Values file and dry render (no change in the cluster).**
   - Write `k8s/monitoring-values.yaml` as in the data model. Run `helm repo add prometheus-community https://prometheus-community.github.io/helm-charts` and `helm repo update`.
   - Record in `docs/runbook.md` the E4.1 findings already measured: `/metrics` answers 200 on port 8080, the four `request_*_seconds` histograms with `model_name="churn"`, and the missing error metric.
   - Check, with `helm template monitoring prometheus-community/kube-prometheus-stack --version 92.22.0 -n monitoring -f k8s/monitoring-values.yaml`:
     - It renders with exit 0.
     - The output has no Alertmanager, node-exporter or kube-state-metrics objects.
     - The `Prometheus` object has `retention: 24h`, a 2Gi `volumeClaimTemplate` with the class `standard`, and both `*SelectorNilUsesHelmValues` flags false.
     - The output has no Grafana object (no Deployment, no Secret `monitoring-grafana`).
   - Check the gates before touching the cluster, with `minikube ssh -- free -m` and `df -h /var`: at least 1.5 GB of available memory and 4 GB of free disk. If the disk is below that, stop and ask the user to free space (for example `docker builder prune`, which is a Docker command and runs on the user's side).

2. **Install the chart.**
   - Run `helm upgrade --install monitoring prometheus-community/kube-prometheus-stack --version 92.22.0 -n monitoring --create-namespace -f k8s/monitoring-values.yaml --wait --timeout 10m`.
   - Record the exact command and the time it took. Helm 4 is new here, so any difference in `--wait` or in the output goes in the gotchas of the runbook.
   - Check:
     - `helm list -n monitoring` shows `monitoring` as `deployed` with chart `kube-prometheus-stack-92.22.0`.
     - Every pod in `monitoring` is `Running` and ready, the PVC of Prometheus is `Bound`, and `kubectl get crd | grep monitoring.coreos.com` lists the CRDs.
     - No pod was `OOMKilled` and none has restarted. `minikube ssh -- free -m` still shows at least 400 MB available.
   - **Stop rule:** if a pod is `Pending`, `OOMKilled` or in a restart loop, or the node starts to thrash (API server timeouts), save `kubectl describe` and the pod logs as evidence, do not retry the install and do not chain restarts, and bring the options to the user (lower limits, raise the node memory, free disk).

3. **Prometheus answers, and Grafana on the Mac reads it (self-scrape only).**
   - Open `kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 19090:9090` in the background.
   - Check:
     - `curl http://localhost:19090/-/ready` answers 200, and the query `up{job="monitoring-kube-prometheus-prometheus"}` returns `1`.
     - `curl http://localhost:3000/api/health` answers `"database": "ok"` (Grafana started with `docker compose -f docker/grafana/compose.yaml up -d`).
     - Grafana, running from `docker/grafana/compose.yaml`, accepts a datasource of type `prometheus` with URL `http://host.docker.internal:19090` (the port-forward seen from the container), and its health check answers `OK`.

4. **Scrape the predictor (E4.3).**
   - Write `k8s/servicemonitor-churn.yaml` and `k8s/networkpolicy-allow-monitoring.yaml` as in the data model and apply both. (The first apply of the `ServiceMonitor` alone left the target `down`; the `NetworkPolicy` was added by decision of the user.)
   - Check:
     - `kubectl get servicemonitor -n monitoring churn-predictor` and `kubectl get networkpolicy -n kubeflow allow-monitoring-scrape` exist.
     - Within 2 minutes, `GET /api/v1/targets` on Prometheus lists a target with `job="churn-predictor"`, `namespace="kubeflow"` and `health="up"`.
   - If the target is missing or `down`, compare the labels of the `ServiceMonitor` with those of the Service and check the port name, then stop and bring the output to the user. Do not edit the Service or the InferenceService to make it fit.

5. **Metrics reach Prometheus.**
   - Open `kubectl port-forward -n kubeflow svc/churn-predictor 18080:80` (not 8080) in the background.
   - Check:
     - The query `request_predict_seconds_count{model_name="churn"}` returns one series with `job="churn-predictor"`. Note its value.
     - Send one raw row in `FEATURES` order to `http://localhost:18080/v1/models/churn:predict` and wait 30 s. The value rises by exactly 1.
     - `histogram_quantile(0.95, rate(request_predict_seconds_bucket{model_name="churn"}[5m]))` returns a number (it can be `NaN` with no traffic; the check is that the query is accepted and the series exist).

6. **Final measurement and settle time.**
   - Wait 10 minutes with the target scraping. Record `free -m`, `df -h /var` and `kubectl get pods -n monitoring` (restarts).
   - Check: no pod restarted, available memory is at least 400 MB, free disk is at least 2 GB, and the PVC of Prometheus uses less than 500 MB (`du` inside the pod, or the `prometheus_tsdb_storage_blocks_bytes` metric).

7. **Docs and hygiene.**
   - Write the "Monitoring" section of `docs/runbook.md`: the commands actually run, the E4.1 findings, the memory and disk numbers before and after, the Compose commands for Grafana, how to reopen the port-forwards, and the gotchas of Helm 4.
   - Add the `DECISIONS.md` log line (chart 92.22.0, Helm 4.3.1, disabled targets, Grafana outside the cluster).
   - Update the README: Prometheus and Grafana rows marked as installed, the risk "Metrics from KServe sklearn server" in §7 marked as solved, and the "Cluster memory" risk with the measured numbers.
   - Tick E4.1 to E4.3 in `BACKLOG.md`.
   - Check: every command in the runbook section appears in the steps above, `grep -rniE 'adminPassword|admin-password' k8s/ docs/ README.md` finds no password value, and the port-forwards are closed.

## Acceptance criteria

### Values and dry render
- [ ] `k8s/monitoring-values.yaml` exists, and `helm template monitoring prometheus-community/kube-prometheus-stack --version 92.22.0 -n monitoring -f k8s/monitoring-values.yaml` exits 0.
- [ ] The rendered output has no object of Alertmanager, node-exporter or kube-state-metrics.
- [ ] The rendered `Prometheus` object has `retention: 24h`, a `volumeClaimTemplate` of 2Gi with the class `standard`, and `serviceMonitorSelectorNilUsesHelmValues` and `podMonitorSelectorNilUsesHelmValues` false.
- [ ] The rendered output has no Grafana object.
- [ ] Before the install, the node had at least 1.5 GB of available memory and 4 GB of free disk, and both numbers are in the runbook.

### E4.1 findings
- [ ] `docs/runbook.md` records that `GET /metrics` on port 8080 of the predictor answers 200, the four `request_*_seconds` histograms with `model_name="churn"`, and that no metric carries an HTTP status.

### Install
- [ ] `helm list -n monitoring` shows the release `monitoring` as `deployed` with the chart `kube-prometheus-stack-92.22.0`.
- [ ] Every pod in `monitoring` is `Running` and ready, and the PVC of Prometheus is `Bound` with 2Gi.
- [ ] `kubectl get crd | grep -c monitoring.coreos.com` prints a number greater than 0.
- [ ] `kubectl get pods -n monitoring` shows 0 restarts, and no pod was `OOMKilled`.
- [ ] `kubectl get pods -A --no-headers | awk '$4!="Running" && $4!="Completed"'` prints nothing.

### Prometheus and Grafana
- [ ] `curl http://localhost:19090/-/ready` answers 200, and the query `up{job="monitoring-kube-prometheus-prometheus"}` returns `1`.
- [ ] `curl http://localhost:3000/api/health` answers with `"database": "ok"` (Grafana on the Mac).
- [ ] Grafana has a datasource of type `prometheus` with URL `http://localhost:19090` whose health check answers `OK`.

### Scrape of the predictor
- [ ] `k8s/servicemonitor-churn.yaml` exists, and `kubectl get servicemonitor -n monitoring churn-predictor` finds it.
- [ ] `k8s/networkpolicy-allow-monitoring.yaml` exists, and `kubectl get networkpolicy -n kubeflow allow-monitoring-scrape` finds it. `default-allow-same-namespace` is unchanged.
- [ ] Within 2 minutes of applying it, `GET /api/v1/targets` on Prometheus lists a target with `job="churn-predictor"`, `namespace="kubeflow"` and `health="up"`.
- [ ] The query `request_predict_seconds_count{model_name="churn"}` returns one series with `job="churn-predictor"`.
- [ ] After one `:predict` request through `svc/churn-predictor` (local port 18080, not 8080) and a wait of 30 s, that value is exactly 1 higher.
- [ ] The query `histogram_quantile(0.95, rate(request_predict_seconds_bucket{model_name="churn"}[5m]))` is accepted, and `request_predict_seconds_bucket` has series in Prometheus.

### Resources after 10 minutes
- [ ] `minikube ssh -- free -m` shows at least 400 MB available, and `df -h /var` shows at least 2 GB free.
- [ ] No pod in `monitoring` has restarted after 10 minutes of scraping.
- [ ] The TSDB of Prometheus uses less than 500 MB.

### No change outside the scope
- [ ] `git diff main -- pipelines components tools docker` is empty (no code and no image change).
- [ ] `kubectl get isvc churn -n kubeflow -o jsonpath='{.metadata.uid}'` has the same value before and after this spec, and the InferenceService and the Service `churn-predictor` have the same `resourceVersion` as before the `ServiceMonitor` was applied.
- [ ] `git diff main -- DECISIONS.md` shows no change to the frozen contracts table, and the log has a dated line for the chart version.
- [ ] `git diff main -- components/tuning_config.json` is empty, and `git ls-files | grep -c '\.csv$'` returns 0.

### Docs and hygiene
- [ ] `docs/runbook.md` has a "Monitoring" section with the commands run, the memory and disk numbers before and after, the Compose commands for Grafana, how to reopen the port-forwards, and the gotchas of Helm 4.
- [ ] `grep -rniE 'adminPassword|admin-password' k8s/ docs/ README.md` finds no password value.
- [ ] The README rows of Prometheus and Grafana say installed, §7 says the risk "Metrics from KServe sklearn server" is solved, and the "Cluster memory" risk carries the measured numbers.
- [ ] `BACKLOG.md` has E4.1 to E4.3 ticked.
- [ ] No `kubectl port-forward` process from this spec is left running.

## Decisions

- **Yes:** split E4 into two specs. SPEC 08 covers E4.1 to E4.3 (metrics reach Prometheus), and SPEC 09 covers E4.4 to E4.6 (load generator, dashboard, submission). E4 did not fit in one objective sentence and touched three areas. The dashboard needs traffic to be checked, so it goes with the load generator.
- **No:** one spec for the whole of E4. It would mix an installation that can fail on memory with a tool and a dashboard that depend on it.
- **Yes:** the stock `sklearnserver` already exposes `/metrics`, measured on 2026-11-07, so no custom predictor is built. The fallback of README §7 is closed.
- **No:** a custom predictor "just in case". It would be a spec of its own and nothing needs it.
- **Yes:** record the missing error metric as a gap and leave the solution to SPEC 09. No metric carries an HTTP status, so the error-rate panel cannot come from the predictor. Covering it (for example counting errors in the load generator) belongs with the load generator.
- **No:** solving the error rate here. It would need a new component or a decision about the panel, and this spec ends at "the target is UP".
- **Yes:** a `ServiceMonitor` on the Service `churn-predictor`. The container declares no named port, and the Service has one (80 → 8080).
- **No:** a `PodMonitor`. It needs a named port, or `portNumber` in an operator version we do not control.
- **No:** the pod annotations `prometheus.kserve.io/*`. The operator does not read them, and `enable-prometheus-scraping` is `"false"` on this pod.
- **Yes:** apply the `ServiceMonitor` by hand from `k8s/` and keep it in `monitoring` with a `namespaceSelector` on `kubeflow`. The scrape then does not depend on every deploy, and everything about monitoring is removed together.
- **No:** having Publish create it. That would change Pipeline 2, which is `Implemented`, for something that does not change with each model version.
- **Yes:** pin the chart to `kube-prometheus-stack` 92.22.0 (app v0.94.2) and name it in the install command, in the runbook and in `DECISIONS.md`. It is the same rule as the image tags: a version that moves can break a working cluster.
- **No:** installing "the latest". The next `helm upgrade` would change the stack without a decision.
- **Yes:** turn off Alertmanager, node-exporter and kube-state-metrics. The node had 2.0 GB of available memory on 2026-11-07, and nothing in the project uses alerts or node metrics.
- **No:** keeping them "because they come by default". Together they cost memory the node does not have.
- **Yes:** turn off the default scrape targets of the cluster (API server, kubelet, control plane, CoreDNS) and the default rules. They add many series and memory, and Prometheus still scrapes itself, which is enough to prove the stack before the predictor target exists.
- **No:** leaving the default targets on to have a richer Grafana. The dashboard of this project only needs the predictor.
- **Yes:** Prometheus keeps 24 h of data on a PVC of 2Gi (StorageClass `standard`). The user asked for persistence. 24 h and one series set are far below 2Gi, and the disk is at 91 %.
- **No:** `emptyDir` for Prometheus. It was the first recommendation, and the user chose the PVC so a restart does not lose the data.
- **No:** a bigger PVC or a longer retention. The disk has 5.2 GB free, and nothing in the project needs history.
- **Yes:** Grafana runs outside the cluster, in a Docker container on the user's Mac (Docker Compose, so nothing is installed on the laptop and the state lives in a named volume), and reads Prometheus through the port-forward `19090`. In the cluster, Grafana 13 took minutes to open its port, the kubelet killed it in a loop (7 restarts in 52 min, also after a longer liveness delay of 300 s) and the node reached a load average of 254 on 12 CPUs, with the API server timing out on 2026-11-09. Memory was not the cause (about 1.9 GB available, no `OOMKilled`).
- **No:** installing Grafana in the cluster with a separate chart or a `startupProbe`. It would not fix the cost of its start on this node. The price of the decision is a step outside the cluster and a datasource created by hand (URL `http://host.docker.internal:19090`).
- **Yes:** no Grafana password is stored anywhere in the repo. The local Grafana asks for a new one at its first login.
- **No:** a password in `k8s/monitoring-values.yaml`. Even for a local cluster, it looks bad in a public repo and a `grep` check cannot tell a fake from a real one.
- **Yes:** `serviceMonitorSelectorNilUsesHelmValues: false` (and the same for pod monitors). Without it, Prometheus only picks up objects with the label of the Helm release, and the `ServiceMonitor` would be ignored without any error.
- **No:** adding the release label to the `ServiceMonitor`. It couples a file in the repo to the name of a Helm release.
- **Yes:** the Helm release is called `monitoring` and lives in the namespace `monitoring`, as README §4 already says. The Service names are then predictable (`monitoring-grafana`, `monitoring-kube-prometheus-prometheus`).
- **Yes:** the port-forwards use local ports 19090 (Prometheus) and 3000 (Grafana, local), and 18080 for the predictor. Port 8080 belongs to the KFP UI, as the spike found.
- **Yes:** memory and disk gates before the install (1.5 GB and 4 GB) and after it (400 MB and 2 GB). The numbers come from the measurement of 2026-11-07, and a heavy install once left this node thrashing (SPEC 03).
- **No:** installing first and looking at the memory afterwards. A thrashing node makes the API server time out and hides the cause.
- **Yes:** a stop rule for the install: if a pod is `Pending`, `OOMKilled` or restarting, save the evidence and bring the options to the user. No retry and no chained restarts.
- **No:** raising the node memory or the PVC inside this spec. It is a decision with side effects on the rest of the cluster, so it goes to the user.
- **Yes:** the verification uses a real request. One `:predict` through the Service must raise `request_predict_seconds_count` by exactly 1. It proves the chain from the pod to the query, and a target that is only `UP` does not.
- **No:** accepting "the target is UP" as the only check. A target can be UP and return no useful series.
- **Yes:** check that the `ServiceMonitor` leaves the InferenceService and the Service untouched (same `uid`, same `resourceVersion`). This spec only adds a monitor, and the criteria make that checkable.
- **Yes:** Helm 4.3.1 is accepted as installed by `brew`. Its differences (`--wait`, output format) are recorded as gotchas in the runbook instead of pinning an older Helm.
- **No:** downgrading Helm. The chart does not require it, and a pinned tool is a second thing to maintain.
- **Yes:** the CRDs stay in the cluster if the release is uninstalled. Helm does not remove them, and the runbook says so.
- **No:** a cleanup script for the stack. The project does not need a teardown path, and the CRDs are shared cluster state.
- **Yes:** Docker cleanup stays with the user. If the disk gate fails, the user runs `docker builder prune` or whichever command fits, because Docker commands are run by the user (`CLAUDE.md`).
- **No:** a change to the frozen contracts. The `FEATURES` order, the registry names and the scikit-learn version are not touched, and the only new line of `DECISIONS.md` is the chart version.
- **Yes:** a `NetworkPolicy` `allow-monitoring-scrape` in `kubeflow`, from the namespace `monitoring` to the predictor pods on TCP 8080. The pre-existing `default-allow-same-namespace` dropped the scrape, and the user approved this option on 2026-11-09.
- **No:** moving Prometheus to `kubeflow` (it mixes monitoring with Kubeflow and breaks the design of this spec), and deleting `default-allow-same-namespace` (it loosens the whole namespace).

## Risks

| Risk | Mitigation |
|---|---|
| The stack does not fit in the node. It had 2.0 GB of available memory (5.8 GB used of 7.8 GB) and 5.2 GB of free disk on 2026-11-07, and the limits of the values add up to about 1.1 GB. A pod is `OOMKilled` or `Pending`, or the node starts to thrash, as it did with MLflow in SPEC 03 (load above 60, API server timeouts). | The memory and disk gates run before the install (step 1) and after it (steps 2 and 6). Alertmanager, node-exporter, kube-state-metrics, the default targets and the default rules are off. The stop rule of step 2 applies: save `kubectl describe` and the logs, no retry and no chained restarts, and bring the options to the user (lower limits, free disk, raise the node memory). Raising the memory is not done inside this spec. |
| The disk fills up while the images are pulled. The node shares its disk with Docker Desktop (91 % used), and the chart pulls several images (Prometheus, the operator, Grafana, its sidecars and the webhook patch job). A full disk makes the kubelet evict pods, including those of KFP and KServe. | The gate of step 1 asks for 4 GB of free disk before the install, and step 6 asks for 2 GB after it. If the gate fails, the user frees space on the Docker side (for example `docker builder prune`, which is a Docker command and runs on the user's side). The 24 h retention and the 2Gi PVC cap what Prometheus itself can take. |
| The `ServiceMonitor` is applied, but Prometheus never creates the target, and nothing reports an error. Prometheus ignores objects without the label of the Helm release unless the selector flags are false. The labels of the Service can also change if KServe is upgraded. | `serviceMonitorSelectorNilUsesHelmValues` and `podMonitorSelectorNilUsesHelmValues` are false in the values, and step 1 checks it in the rendered `Prometheus` object. The selector uses the two labels read from the Service on 2026-11-07. Step 4 lists the targets of the API, and if the target is missing it compares labels and port name and stops, without editing the Service. |
| The target is `UP` but returns no useful series, or the metric names change with a KServe version. | Step 5 sends one real request and requires `request_predict_seconds_count` to rise by exactly 1. The runbook records the metric names found on 2026-11-07, and SPEC 09 builds the dashboard from those same names. |
| The predictor pod is replaced (a Publish run, or a restart) and the series start again from 0, so `rate()` shows a reset or a gap. | The target is the Service, so Prometheus follows the new pod without a change. `rate()` and `histogram_quantile` handle counter resets. SPEC 09 reads the dashboard during a load run, with no pod change. |
| Helm 4 behaves differently from what the documentation of the chart assumes: `--wait`, `--atomic`, the server-side apply default or the output format. The first install fails or hangs in an unclear state. | The command of step 2 is run once and recorded with its output. If it fails, the release state (`helm list -n monitoring -a`) and the events are saved, and the options go to the user before any `helm uninstall` or retry. Gotchas go in the runbook. |
| The chart's CRDs and the admission webhook cause trouble: a half-installed release leaves CRDs behind, and a later install fails on webhook certificates or on stale objects. | The CRDs are cluster-level and Helm does not remove them on uninstall, and the runbook says so. If an install fails halfway, the state is inspected first. The CRDs are not deleted inside this spec without a decision from the user. |
| The Prometheus PVC uses the StorageClass `standard` (hostPath, reclaim policy `Delete`) on the same node disk. If the PVC is deleted or the node is recreated, the data is lost. | The data is only 24 h of a learning project, and the user accepted the PVC for restarts, not for durability. The runbook says the PVC does not survive `minikube delete`. |
| The Grafana password leaks. | Grafana is local and not in the repo. The final `grep` of the criteria still checks that no password value is in `k8s/`, `docs/` or the README. |
| A port-forward is left running and holds a local port, or one collides with the KFP UI on 8080. | The plan uses only 19090, 13000 and 18080. The last criterion checks that no port-forward of this spec is still running. |
| KFP runs fail in their first minutes (the drivers time out against `metadata-grpc`; the cause was never found) while the new stack uses the node's memory and CPU. The flake could be blamed on, or made worse by, the stack. | This spec does not run any pipeline. If a later run fails with `dial tcp ... i/o timeout`, the stack is checked against the runbook measurements first. The criteria require the same `uid` and `resourceVersion` for the InferenceService, so nothing in Pipeline 2 changes. |

## What is **not** in this spec

- The Grafana dashboard, its JSON, the screenshot and the choice of how to cover the error rate (SPEC 09, E4.5).
- The load generator, `submission.csv` and its check (SPEC 09, E4.4 and E4.6).
- A custom predictor. The stock `sklearnserver` already exposes `/metrics`.
- Alertmanager, alert rules, node-exporter and kube-state-metrics.
- Scraping anything other than the `churn` predictor (KFP, MLflow, the KServe controller).
- Long retention, remote storage, high availability, ingress and authentication for Prometheus or Grafana.
- Any change to Publish, to the InferenceService, to Pipeline 1 or to Pipeline 2.
- Raising the node memory or resizing the Docker Desktop disk. If the stack does not fit, that is a decision brought to the user.
- Cleaning Docker images or build cache. If the disk check needs it, the user runs it.
- The full 30-trial run, still deferred to the end of the project.

Each one of those, if it lands, goes in its own spec.
