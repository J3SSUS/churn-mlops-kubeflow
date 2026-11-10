# SPEC 09 — Load generator, provisioned dashboard and submission

> **Status:** Implemented
> **Depends on:** SPEC 01, SPEC 02, SPEC 03, SPEC 04, SPEC 05, SPEC 06, SPEC 07, SPEC 08
> **Backlog:** E4 (Monitoring and load generator), tasks E4.4 to E4.6 of `BACKLOG.md`
> **Date:** 2026-11-09
> **Objective:** Add a load generator that replays `test.csv` against the live `churn` endpoint, writes a checked `submission.csv` and exposes its own request metrics, so that a Grafana dashboard provisioned from the repo shows request rate, p50/p95 latency and error rate.

## Scope

**In:**

- **E4.4 Load generator:**
  - `tools/load_generator.py` reads `test.csv` from the PVC (`RAW_DIR_PVC` of `config.py`), builds each row in `FEATURES` order (numbers as floats, strings as strings, the same payload as the smoke test) and posts batches of 100 rows, one request at a time, at 2 requests per second, to `http://churn-predictor.kubeflow.svc.cluster.local/v1/models/churn:predict`.
  - It makes `PASSES` passes over the file (default 5, about 4 minutes). Only the first pass keeps the predictions and writes `submission.csv` (`id,Exited`, the second column of `predictions`) to the PVC. The other passes are traffic for the dashboard.
  - A failed request is retried 3 times with a pause. Every failed attempt counts as an error. If a batch still fails, the Job exits with an error and no incomplete `submission.csv` is written.
  - It serves its own `/metrics` on a named port while it runs (standard library only): `load_requests_total{result="ok|error"}`.
  - `k8s/load-generator-job.yaml`, a Job in `kubeflow` with `churn-base:0.3.0`, the PVC `churn-data`, and the script and `config.py` mounted from ConfigMaps (no image rebuild). The ConfigMap of the script is created by hand with `kubectl`.
- **E4.5 Dashboard:**
  - `k8s/load-generator-podmonitor.yaml`, a `PodMonitor` in `monitoring` for the port of the Job pod.
  - A second `NetworkPolicy` document in `k8s/networkpolicy-allow-monitoring.yaml` so Prometheus can reach that port (the namespace `kubeflow` drops ingress from `monitoring`, found in SPEC 08).
  - `k8s/grafana-dashboard.json` with four panels: request rate and p50 and p95 latency from `request_predict_seconds` (predictor side), and error rate from `load_requests_total` (client side).
  - Grafana provisioned from the repo: a datasource file with a fixed `uid` (URL `http://host.docker.internal:19090`) and a dashboards file in `docker/grafana/provisioning/`. `docker/grafana/compose.yaml` mounts them and the JSON, so the dashboard appears when the container starts.
  - A screenshot of the dashboard during a load run, taken by the user, saved in `docs/img/`.
- **E4.6 Submission check:** `tools/check_submission.py`, run on the laptop against the `submission.csv` copied from the PVC with `kubectl cp`. It checks 10,000 rows, the same ids as `data/sample_submission.csv`, and values in [0, 1] with no NaN.
- **Docs:** a section in `docs/runbook.md` with the commands run and the gotchas, dated lines in `DECISIONS.md` (client-side error rate, provisioned Grafana), README rows and the error-rate risk marked as solved, and E4.4 to E4.6 ticked in `BACKLOG.md` once the user agrees.

**Out of scope (for future specs):**

- Submitting `submission.csv` to Kaggle and its score (the user does it by hand, and the score goes into the README results of E5.2).
- An error metric on the predictor side (custom predictor or Pushgateway). The error rate of this spec is client-side and exists only while the generator runs.
- Alerts, Alertmanager, stress tests, concurrent load and autoscaling.
- Running the generator from a KFP pipeline. It is a standalone Job.
- Measuring the quality of the predictions. `test.csv` has no labels, and the score comes from Kaggle.
- More dashboards than the one of this spec, and Grafana back in the cluster.
- The runbook and README polish of E5.1 and E5.2, beyond the parts this spec touches.
- The 30-trial training run and the second deploy run, which stay owed until the end of the project.

## Data model

This spec adds no business data and no frozen contract changes. The `FEATURES` order, the target and the registry names stay as in `DECISIONS.md`. It defines the files, a few shared constants, the request and the file the generator produces, the metric it serves, the Kubernetes objects and the dashboard.

### Files added or changed

| File | Change | Purpose |
|---|---|---|
| `tools/load_generator.py` | new | The generator. Runs in the Job (below) |
| `tools/check_submission.py` | new | Local check of `submission.csv` |
| `k8s/load-generator-job.yaml` | new | The Job |
| `k8s/load-generator-podmonitor.yaml` | new | Prometheus scrapes the Job pod |
| `k8s/networkpolicy-allow-monitoring.yaml` | second document added | Lets Prometheus reach port 9100 of the Job pod |
| `k8s/grafana-dashboard.json` | new | The dashboard |
| `docker/grafana/provisioning/datasources/prometheus.yaml` | new | Datasource with a fixed `uid` |
| `docker/grafana/provisioning/dashboards/dashboards.yaml` | new | File provider for the dashboard |
| `docker/grafana/compose.yaml` | changed | Mounts the provisioning files and the dashboard JSON |
| `components/config.py` | constants added | Shared defaults (below). The ConfigMap `churn-code` is refreshed with `tools/sync_code.sh` |

### Constants added to `components/config.py`

```python
SUBMISSION_PATH = "/mnt/data/submission.csv"   # on the PVC, written by the load generator
LOAD_BATCH_SIZE = 100                          # rows per request
LOAD_RATE_RPS = 2.0                            # requests per second, one at a time
LOAD_PASSES = 5                                # passes over test.csv; only the first keeps predictions
LOAD_RETRIES = 3                               # attempts per batch after the first one
LOAD_RETRY_PAUSE_S = 2.0                       # pause between attempts of the same batch
LOAD_METRICS_PORT = 9100                       # /metrics of the generator
```

The script takes them as defaults of its arguments (`--batch-size`, `--rate`, `--passes`, `--retries`, `--retry-pause`), so the Job can override them without a code change. It also takes `--url`, `--data` and `--output`, so a local test can point elsewhere.

### Request and answer (same contract as the smoke test)

```
POST http://churn-predictor.kubeflow.svc.cluster.local/v1/models/churn:predict

request:  {"instances": [[CreditScore, Geography, Gender, Age, Tenure, Balance,
                          NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary], ...]}
          # 100 rows, FEATURES order, numbers as floats and strings as strings
answer:   {"predictions": [[p0, p1], ...]}
          # one pair per row, in order
```

The churn probability is the second column. `id`, `CustomerId` and `Surname` are never sent; `id` is kept aside to build the file. An attempt counts as an error if the status is not 200, the connection fails or times out, or `predictions` does not have one pair of finite values in [0, 1] per row.

### `submission.csv` on the PVC

```
id,Exited
15000,0.5072928468077883
...                                  # 10,000 rows, ids in the order of test.csv, Exited = p1
```

Written to `/mnt/data/submission.csv` through a temporary file and a rename, only when the first pass has finished with every batch answered. A failed run leaves no file (and an older one is deleted at the start, so a stale file is never mistaken for a new one).

### Metric served by the generator

Standard library only, text exposition format, port `9100` (named `metrics`) while the Job runs. Both series exist from the start, at 0, so `rate()` works from the first scrape:

```
# TYPE load_requests_total counter
load_requests_total{result="ok"} 0
load_requests_total{result="error"} 0
```

It counts HTTP attempts, so a retried batch adds one `error` and then one `ok`. It is a client-side error rate: it exists only while the generator runs, and a Job that finished leaves no target.

### `k8s/load-generator-job.yaml` (shape)

```yaml
apiVersion: batch/v1
kind: Job
metadata: {name: load-generator, namespace: kubeflow}
spec:
  backoffLimit: 0                 # a failed run is looked at, not repeated
  activeDeadlineSeconds: 900
  template:
    metadata: {labels: {app: load-generator}}
    spec:
      restartPolicy: Never
      containers:
        - name: generator
          image: churn-base:0.3.0
          imagePullPolicy: IfNotPresent
          command: ["python", "/generator/load_generator.py"]
          env:
            - {name: PYTHONPATH, value: /code}
            - {name: OMP_NUM_THREADS, value: "1"}
          ports: [{name: metrics, containerPort: 9100}]
          resources: {requests: {cpu: 100m, memory: 256Mi}, limits: {memory: 512Mi}}
          volumeMounts:
            - {name: data, mountPath: /mnt/data}
            - {name: code, mountPath: /code}               # ConfigMap churn-code
            - {name: generator, mountPath: /generator}     # ConfigMap load-generator
```

The ConfigMap `load-generator` holds `load_generator.py` and is created by hand with `kubectl create configmap load-generator -n kubeflow --from-file=load_generator.py=tools/load_generator.py --dry-run=client -o yaml | kubectl apply -f -`.

### `k8s/load-generator-podmonitor.yaml` and the second NetworkPolicy

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata: {name: load-generator, namespace: monitoring}
spec:
  namespaceSelector: {matchNames: [kubeflow]}
  selector: {matchLabels: {app: load-generator}}
  podMetricsEndpoints:
    - {port: metrics, path: /metrics, interval: 15s}
---
# second document of k8s/networkpolicy-allow-monitoring.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: allow-monitoring-scrape-load-generator, namespace: kubeflow}
spec:
  podSelector: {matchLabels: {app: load-generator}}
  policyTypes: [Ingress]
  ingress:
    - from: [{namespaceSelector: {matchLabels: {kubernetes.io/metadata.name: monitoring}}}]
      ports: [{protocol: TCP, port: 9100}]
```

### Dashboard `k8s/grafana-dashboard.json`

`uid` `churn-serving`, title "Churn serving", last 15 minutes, refresh 10 s. Every panel points at the datasource `uid` `prometheus-local`. Four panels:

| Panel | PromQL | Side |
|---|---|---|
| Request rate (req/s) | `sum(rate(request_predict_seconds_count{model_name="churn"}[1m]))` | predictor |
| Latency p50 | `histogram_quantile(0.5, sum by (le) (rate(request_predict_seconds_bucket{model_name="churn"}[1m])))` | predictor |
| Latency p95 | `histogram_quantile(0.95, sum by (le) (rate(request_predict_seconds_bucket{model_name="churn"}[1m])))` | predictor |
| Error rate (%) | `100 * sum(rate(load_requests_total{result="error"}[1m])) / sum(rate(load_requests_total[1m]))` | generator |

Without traffic the latency panels show no data (the series of the predictor exist only after its first request, found in SPEC 08), and the error panel shows no data when no Job is running. The title of the error panel says it is a client-side rate.

### Grafana provisioning

```yaml
# docker/grafana/provisioning/datasources/prometheus.yaml
apiVersion: 1
datasources:
  - {name: "Prometheus (repo)", uid: prometheus-local, type: prometheus, access: proxy,
     url: "http://host.docker.internal:19090", isDefault: false, editable: false}

# docker/grafana/provisioning/dashboards/dashboards.yaml
apiVersion: 1
providers:
  - {name: churn, folder: Churn, type: file, disableDeletion: true, options: {path: /etc/grafana/dashboards}}
```

`docker/grafana/compose.yaml` adds two read-only mounts: `./provisioning` to `/etc/grafana/provisioning`, and `../../k8s/grafana-dashboard.json` to `/etc/grafana/dashboards/churn-serving.json`. The datasource keeps a different name from the one created by hand in SPEC 08, so the two do not collide; the old one can be deleted from the UI.

### `tools/check_submission.py`

`uv run python tools/check_submission.py <submission.csv> [--sample data/sample_submission.csv]`. It exits 0 and prints one `RESULT ok` line only if: the header is exactly `id,Exited`; there are 10,000 rows; the ids equal those of the sample file, in the same order; and every `Exited` is a number in [0, 1] with no NaN. Any other case prints the first violation and exits 1.

### Objects and layout

| Object | Namespace | Notes |
|---|---|---|
| ConfigMap `load-generator` | `kubeflow` | The script, created by hand |
| Job `load-generator` | `kubeflow` | Deleted and re-created for each run (a Job cannot be re-run) |
| PodMonitor `load-generator` | `monitoring` | |
| NetworkPolicy `allow-monitoring-scrape-load-generator` | `kubeflow` | Added to the existing file |
| `/mnt/data/submission.csv` | PVC `churn-data` | New file. `raw/`, `processed/` and `models/` are untouched |

## Implementation plan

Stop rule for every step that touches the cluster: before a Job, `minikube ssh -- uptime` shows a 1-minute load below 20 and `free -m` at least 1 GB available. If a pod is `Pending`, `OOMKilled` or restarting, or the API server times out, save `kubectl describe` and the logs as evidence, do not retry, and bring the options to the user (SPEC 08 had the same rule and needed it).

1. **Decisions and assumptions (no code).**
   - Add two dated lines to `DECISIONS.md`: the error rate is client-side (the generator serves `load_requests_total`, the predictor has no status metric), and Grafana is provisioned from the repo.
   - Check on the cluster with the helper pod `k8s/data-loader-pod.yaml` (apply, look, delete): `/mnt/data/raw/test.csv` exists and has 10,000 rows plus the header, and `/mnt/data/submission.csv` does not exist yet.
   - Check from a pod in `kubeflow` with `churn-base:0.3.0` that `pandas` imports and that `POST` to the endpoint of the Service answers 200 for the first row of `test.csv`.
   - Check: the two log lines exist, and both facts about the PVC and the endpoint are written in the runbook.

2. **Write `tools/load_generator.py` and the constants (local checks only).**
   - Add the constants to `components/config.py` (the data model lists them, including `LOAD_RETRY_PAUSE_S`).
   - Write the generator with the arguments `--url`, `--batch-size`, `--rate`, `--passes`, `--retries`, `--retry-pause`, `--data` and `--output` (so a local test can point elsewhere).
   - Check with a throwaway HTTP server on the laptop (as in SPEC 07), recording the commands in the runbook:
     - Every request has 100 rows in `FEATURES` order, numbers as floats and strings as strings, and none carries `id`, `CustomerId` or `Surname`.
     - One pass over a 250-row file sends 3 requests (100, 100, 50), and the rate is held within 10 % of the target.
     - The file written has `id,Exited`, the ids in the order of the input and the second column of the answer. The throwaway server answers with a value that depends on the row, so the order of the answers is proven.
     - A server that fails twice and then answers: the batch succeeds, and `load_requests_total` shows 2 `error` and the `ok` attempts. A server that always fails: the process exits non-zero after the retries and no `submission.csv` exists. A stale `submission.csv` is gone after that failed run.
     - A malformed answer (wrong number of rows, a value of 1.5, a `NaN`) counts as an error.
     - `GET /metrics` on the generator returns both series from the start, at 0.

3. **Write `tools/check_submission.py` (local).**
   - Check with `data/sample_submission.csv` as the good case (it passes), and with copies broken one at a time in the scratchpad: a missing row, a changed id, a swapped order, a `NaN`, a value of 1.2, a wrong header. Each bad case exits 1 and names the first violation.

4. **Run the generator once on the cluster (one pass, no dashboard yet).**
   - Run `tools/sync_code.sh`, create the ConfigMap `load-generator`, write `k8s/load-generator-job.yaml` and apply it with `--passes 1`.
   - Check:
     - The Job `Complete`s, its log shows the 100 requests, the achieved rate, the latency of a batch and no error, and the pod ended with `reason: Completed`.
     - `/mnt/data/submission.csv` exists (helper pod, then deleted) and is copied to `data/submission.csv` (git-ignored) with `kubectl cp`.
     - `tools/check_submission.py data/submission.csv` prints `RESULT ok`.
     - `request_predict_seconds_count` rose by exactly 100 in Prometheus.

5. **Make Prometheus scrape the generator.**
   - Write `k8s/load-generator-podmonitor.yaml`, add the second document to `k8s/networkpolicy-allow-monitoring.yaml`, and apply both.
   - Delete and re-create the Job with `--passes 2`. While it runs:
     - Prometheus lists a target of the `load-generator` pod with `health="up"` within 30 s of the pod starting.
     - `load_requests_total{result="ok"}` rises and `result="error"` stays at 0.
   - Then run a Job against a model name that does not exist (`--url .../v1/models/nope:predict --retry-pause 10`): `result="error"` rises in Prometheus before the Job fails, the Job ends `Failed`, and `submission.csv` is gone from the PVC. Run the successful Job again afterwards so the file exists.

6. **Write the dashboard and the provisioning.**
   - Write `k8s/grafana-dashboard.json`, the two provisioning files and the new mounts of `docker/grafana/compose.yaml`.
   - Check, with the port-forward `19090` open: the JSON parses, has the `uid` `churn-serving` and four panels that all use the datasource `prometheus-local`, and each of the four PromQL expressions is accepted by `GET /api/v1/query` (status `success`, the result can be empty).
   - The user runs `docker compose -f docker/grafana/compose.yaml up -d`, which recreates the container. Check: the folder "Churn" holds the dashboard "Churn serving", the datasource "Prometheus (repo)" answers `OK`, and `docker compose logs grafana` has no provisioning error.

7. **Full run with the dashboard open.**
   - With the port-forward `19090` and Grafana open, delete and re-create the Job with the defaults (5 passes, about 4 minutes).
   - Check: the four panels show data during the run (request rate near 2 req/s, p50 and p95 with numbers, error rate at 0 %), the Job `Complete`s, and the new `submission.csv` passes `tools/check_submission.py`.
   - The user takes the screenshot and saves it as `docs/img/grafana-churn-serving.png`.
   - After the run: no restart in `monitoring`, load average back under 20.

8. **Docs and hygiene.**
   - Write the section of `docs/runbook.md`: the commands run, how to run the generator and re-create the Job, how to copy and check the submission, the provisioning commands, and the gotchas found.
   - Update the README: the load generator and the dashboard as installed, the risk "Metrics from KServe sklearn server" noted as covered by the client-side metric, and the repo layout.
   - Tick E4.4 to E4.6 in `BACKLOG.md` once the user agrees.
   - Check: `git status` shows no `data/` file and no `submission.csv` to commit, `grep -rniE 'adminPassword|admin-password' k8s/ docker/ docs/ README.md` finds no password value, the helper pod and the Jobs are deleted, and no `kubectl port-forward` of this spec is left running.

## Acceptance criteria

### Decisions and assumptions
- [x] `DECISIONS.md` has two dated lines for this spec (client-side error rate, Grafana provisioned from the repo), and `git diff main -- DECISIONS.md` shows no change to the frozen contracts table.
- [x] `docs/runbook.md` records that `/mnt/data/raw/test.csv` has 10,000 rows plus the header, that `pandas` imports in `churn-base:0.3.0`, and that the endpoint of the Service answered 200 for a row of `test.csv`.

### Load generator (local, throwaway HTTP server)
- [x] `components/config.py` has the seven constants of the data model (`SUBMISSION_PATH`, `LOAD_BATCH_SIZE`, `LOAD_RATE_RPS`, `LOAD_PASSES`, `LOAD_RETRIES`, `LOAD_RETRY_PAUSE_S`, `LOAD_METRICS_PORT`), and `git diff main -- components/config.py` only adds lines.
- [x] Every request has at most 100 rows in `FEATURES` order, numbers as floats and strings as strings, and none carries `id`, `CustomerId` or `Surname`.
- [x] One pass over a 250-row file sends 3 requests (100, 100, 50 rows), and the measured rate is within 10 % of `--rate`.
- [x] The file written has the header `id,Exited`, the ids in the order of the input and the second column of each answer.
- [x] A server that fails twice and then answers: the batch succeeds, and `load_requests_total` shows 2 `error` and the `ok` attempts.
- [x] A server that always fails: the process exits non-zero after the retries, no `submission.csv` exists, and a stale one that was there before is gone.
- [x] An answer with the wrong number of rows, a value of 1.5 or a `NaN` counts as an error.
- [x] `GET /metrics` on the generator returns both series, `result="ok"` and `result="error"`, at 0 before the first request.
- [x] No dependency was added: `git diff main -- pyproject.toml uv.lock docker/churn-base` is empty.

### Submission check (local)
- [x] `tools/check_submission.py data/sample_submission.csv` exits 0 and prints `RESULT ok`.
- [x] Six copies broken one at a time (a missing row, a changed id, two ids swapped, a `NaN`, a value of 1.2, a wrong header) each exit 1 and name the first violation.

### Generator on the cluster (one pass)
- [x] `kubectl get configmap -n kubeflow load-generator` exists, and `k8s/load-generator-job.yaml` is applied with `--passes 1`.
- [x] The Job is `Complete`, its pod ended with `reason: Completed` (not `OOMKilled`), and its log shows 100 requests and no error.
- [x] `/mnt/data/submission.csv` exists on the PVC, `kubectl cp` copies it to `data/submission.csv`, and `tools/check_submission.py data/submission.csv` prints `RESULT ok`.
- [x] `request_predict_seconds_count{model_name="churn"}` is exactly 100 higher than before the Job, after waiting 30 s.

### Prometheus scrapes the generator
- [x] `kubectl get podmonitor -n monitoring load-generator` and `kubectl get networkpolicy -n kubeflow allow-monitoring-scrape-load-generator` exist, and `allow-monitoring-scrape` and `default-allow-same-namespace` are unchanged.
- [x] During a 2-pass run, `GET /api/v1/targets` lists a target of the pod `load-generator` with `health="up"` within 30 s of the pod starting, `load_requests_total{result="ok"}` rises, and `result="error"` stays at 0.
- [x] A Job against `.../v1/models/nope:predict` makes `load_requests_total{result="error"}` rise in Prometheus before it ends, ends `Failed`, and leaves no `/mnt/data/submission.csv`.
- [x] After that, a successful Job leaves a `submission.csv` that passes `tools/check_submission.py`.

### Dashboard and provisioning
- [x] `k8s/grafana-dashboard.json` parses, has the `uid` `churn-serving` and four panels, and every panel uses the datasource `uid` `prometheus-local`.
- [x] Each of the four PromQL expressions of the dashboard is accepted by `GET /api/v1/query` on Prometheus (`status` `success`).
- [x] `docker/grafana/provisioning/datasources/prometheus.yaml`, `docker/grafana/provisioning/dashboards/dashboards.yaml` and the two new mounts of `docker/grafana/compose.yaml` exist, and `docker compose -f docker/grafana/compose.yaml config` (run by the user) exits 0.
- [x] After the user recreates the container, the folder "Churn" holds the dashboard "Churn serving", the datasource "Prometheus (repo)" answers `OK`, and `docker compose logs grafana` shows no provisioning error, apart from the two harmless lines about the optional folders `provisioning/plugins` and `provisioning/alerting`, which do not exist because the mount replaces the default folder.

### Full run with the dashboard
- [x] During a run with the defaults (5 passes), the four panels show data: request rate averaging between 1.5 and 2 req/s over the run (measured 1.68: the generator is sequential and its slow batches keep it a little under the target of 2, with single passes at 1.44 to 1.83 and one-minute windows down to 0.9), p50 and p95 with numbers, and error rate at 0 %.
- [x] That Job is `Complete`, and its `submission.csv` passes `tools/check_submission.py`.
- [x] `docs/img/grafana-churn-serving.png` exists and shows the four panels with data (checked by the user).
- [x] After the run, no pod in `monitoring` has restarted since before the run, and `minikube ssh -- uptime` shows a 1-minute load below 20.

### No change outside the scope
- [x] `git diff main -- components pipelines ':!components/config.py'` is empty, and `git diff main -- components/tuning_config.json` is empty.
- [x] `kubectl get isvc churn -n kubeflow -o jsonpath='{.metadata.uid}'` still prints `6519f4b2-2a3e-4286-a18c-4479f2075584`, and nothing in this spec edits the InferenceService or the Service `churn-predictor`.
- [x] `git ls-files | grep -c '\.csv$'` returns 0, `git ls-files | grep -c 'submission.*\.csv$'` returns 0 (the name `check_submission.py` and this spec contain the word, they are not data), and `git check-ignore data/submission.csv` prints the path.

### Docs and hygiene
- [x] `docs/runbook.md` has a section with the commands run, how to run and re-create the Job, how to copy and check the submission, the provisioning commands and the gotchas found.
- [x] The README says the load generator and the dashboard are installed, notes that the error rate is covered by the client-side metric, and the repo layout lists the new files.
- [x] `BACKLOG.md` has E4.4 to E4.6 ticked.
- [x] `grep -rniE 'adminPassword|admin-password' k8s/ docker/ docs/ README.md` finds no password value.
- [x] The helper pod and the Jobs of this spec are deleted, and no `kubectl port-forward` of this spec is left running.

## Decisions

### Error rate
- **Yes:** the error rate comes from the generator. It serves `load_requests_total{result="ok|error"}` on port 9100 with the standard library, and a `PodMonitor` makes Prometheus scrape it. The predictor has no metric with an HTTP status (measured in SPEC 08), and this is the option that puts real data in the panel without a new component.
- **No:** dropping the error-rate panel. The README and `BACKLOG.md` ask for it.
- **No:** Pushgateway or a custom predictor that exports statuses. Another component or another image on a node that already overloaded once (SPEC 08), for a number that only matters while the generator runs.
- **Yes:** accept the limit. The panel is a client-side rate, exists only while a Job runs, and shows no data otherwise. The runbook and the title of the panel say so.
- **Yes:** `PodMonitor`, not `ServiceMonitor`: the Job pod has a named port and no Service.
- **Yes:** a second `NetworkPolicy` document for port 9100 of the generator pods. **No:** widening `allow-monitoring-scrape`, which selects the predictor labels and would open its port 8080 to nothing new.

### Generator
- **Yes:** the script travels in the ConfigMap `load-generator`, with `churn-base:0.3.0` and the shared code from `churn-code`. It is the pattern of SPEC 04 and SPEC 07 and needs no image rebuild. **No:** a new image `churn-load-generator`.
- **Yes:** one request at a time, 100 rows per request, 2 requests per second, 5 passes (about 4 minutes). The node had a load average of 254 once, and nothing here needs concurrency. **No:** parallel workers.
- **Yes:** only the first pass keeps predictions and writes `submission.csv`. The other passes are traffic for the dashboard. The file then comes from one complete pass over the served model.
- **Yes:** 3 retries per batch with a pause, every failed attempt counted as an error. If a batch still fails, the Job fails and no file is written, and a stale file is deleted at the start. **No:** writing a partial `submission.csv`, and **no:** `backoffLimit` above 0, because a failed run is looked at, not repeated.
- **Yes:** one Job per run, deleted and re-created, because a Job cannot be re-run.
- **Yes:** the defaults live in `components/config.py` and are the defaults of the script arguments, so the Job can override them. `tools/sync_code.sh` refreshes `churn-code`.
- **Yes:** the generator takes the endpoint from the Service name (`http://churn-predictor.kubeflow.svc.cluster.local/...`), with no port-forward. No part of the generator needs one.

### Dashboard
- **Yes:** the dashboard and the datasource are provisioned from the repo, so the dashboard appears when the container starts (the user's choice). **No:** importing the JSON by hand.
- **Yes:** the JSON is written by hand and reviewed in the diff, and lives at `k8s/grafana-dashboard.json` (the path of `BACKLOG.md`), mounted into the container from there. **No:** building it in the UI and exporting it, which leaves an unreviewable blob and a random `uid`.
- **Yes:** the datasource has a fixed `uid` (`prometheus-local`) and the name "Prometheus (repo)", different from the one made by hand in SPEC 08, so they do not collide. **No:** a datasource variable in the JSON.
- **Yes:** `[1m]` windows with the 15 s scrape interval of SPEC 08. **No:** longer windows, which would hide a 4-minute run.
- **Yes:** the price is accepted: `docker/grafana/compose.yaml` of SPEC 08 changes.

### Submission check
- **Yes:** a local script that compares the file with `data/sample_submission.csv`, after `kubectl cp` to `data/submission.csv`. Both files are data and stay git-ignored. **No:** checking inside the Job, which would need the sample file on the PVC.
- **Yes:** Kaggle submission and its score are out of this spec. The user does it, and the score goes into the README results of E5.2.

### Scope guards
- **Yes:** the spec edits no predictor object. The InferenceService and the Service `churn-predictor` are untouched (same `uid`), and no component or pipeline changes other than the constants of `config.py`.
- **Yes:** the node gate of SPEC 08 repeats before every Job (load below 20, at least 1 GB available), because the generator runs on the same tight node.
- **No:** changes to the frozen contracts. The `FEATURES` order, the target and the registry names stay as they are.

## Risks

| Risk | Mitigation |
|---|---|
| The generator overloads the node. It runs next to the predictor, Prometheus and KFP on the same node that reached a load average of 254 in SPEC 08. | One request at a time, 100 rows, 2 requests per second, `requests` and a memory `limit` on the pod, `activeDeadlineSeconds: 900`. The node gate (load below 20, at least 1 GB available) runs before every Job, and the stop rule of the plan saves evidence and stops on `Pending`, `OOMKilled` or API timeouts. |
| The target rate cannot be held. The generator is sequential, so if one 100-row request takes more than 0.5 s the real rate is below 2 per second, and the request-rate panel shows less than expected. The latency of a 100-row batch is not measured yet (the smoke test sends 3 rows). | The log of the one-pass run (step 4) reports the achieved rate and the latency of a batch. If the target is out of reach, `--rate` or `--batch-size` are lowered, and the new defaults go into `config.py` and the runbook. The 10 % check of the criteria applies to the local test, not to the cluster. |
| The panels are empty or show gaps. The series of the predictor exist only after its first request (found in SPEC 08), `rate()` needs two samples, and the error panel has no data without a running Job. | The dashboard is meant to be read during a run. The runbook says so, the error panel is titled as a client-side rate, and the 15-minute window keeps the data of the last run visible. |
| Prometheus misses a short Job. The scrape interval is 15 s, so a Job that ends in a few seconds can leave no sample, and the check of the error run would be flaky. | The run against a model that does not exist uses `--retry-pause 10`, so it lasts at least 30 s. The criteria ask for the rise of `result="error"` before the Job ends. |
| Grafana shows errors instead of data. The datasource goes through the port-forward `19090`, which dies with the terminal or with `minikube stop`, and `host.docker.internal` must resolve from the container. | The runbook has the commands to reopen the port-forward and to recreate the container. The datasource health check of the criteria is the first thing looked at when a panel fails. |
| The dashboard in Grafana and the one in the repo drift apart. The mounted file is read-only, and a change made in the UI is not saved to it. A single-file bind mount can also keep serving the old content after an editor replaces the file. | The JSON in the repo is the only source, and the runbook says to edit it there and then restart the container. |
| The submission is correct in format but wrong in order or stale. The check compares ids and ranges, not whether row N got the prediction of row N, and the file belongs to the model version served at that moment (today `hist_gradient_boosting`, which the 30-trial run at the end of the project can replace). | The throwaway server of the local test answers with a value that depends on the row, so the order of the answers is proven. The generator deletes an old file at the start. The runbook records the `storageUri` of the InferenceService at the time of the run, and the submission is generated again after the last deploy. |
| The exact `+100` in `request_predict_seconds_count` breaks if anything else queries the predictor during the run. | The step runs with no other client. If the number is off, the first thing checked is `kubectl get pods` and the log of the Job. |
