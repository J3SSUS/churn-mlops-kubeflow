"""Load generator (SPEC 09, E4.4): replay test.csv against the live endpoint, write submission.csv
and serve its own request metrics.

It posts batches of rows, one request at a time, at a fixed rate. The first pass over the file keeps
the probabilities and writes `submission.csv` (`id,Exited`); the other passes only produce traffic for
the dashboard. A failed attempt is retried, and every failed attempt counts as an error in
`load_requests_total{result="error"}`. If a batch still fails, the run exits with 1 and no
incomplete `submission.csv` is left behind.

Runs in a Job (PYTHONPATH=/code gives `config`) and locally with PYTHONPATH=components:tools.
Standard library only.

    python tools/load_generator.py --data data/test.csv --output /tmp/submission.csv --url http://...
"""
import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config

DEFAULT_URL = (f"http://{config.ISVC_NAME}-predictor.{config.K8S_NAMESPACE}.svc.cluster.local"
               f"/v1/models/{config.ISVC_NAME}:predict")
REQUEST_TIMEOUT_S = 60


class Counters:
    """`load_requests_total` by result. Both series exist from the start, at 0."""

    def __init__(self):
        self._lock = threading.Lock()
        self.values = {"ok": 0, "error": 0}

    def add(self, result):
        with self._lock:
            self.values[result] += 1

    def render(self):
        with self._lock:
            lines = ["# HELP load_requests_total HTTP attempts of the load generator, by result.",
                     "# TYPE load_requests_total counter"]
            lines += [f'load_requests_total{{result="{r}"}} {v}' for r, v in self.values.items()]
        return "\n".join(lines) + "\n"


def start_metrics_server(counters, port):
    """Serve GET /metrics on a daemon thread. Returns the server (its port is in server_address)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send_error(404)
                return
            body = counters.render().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def read_rows(path):
    """(ids, instances): the ids kept aside, and each row in FEATURES order with numbers as floats
    and strings as strings, as a client sends them. The columns are picked by name, once."""
    ids, instances = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row[config.ID_COL])
            instances.append([float(row[c]) if i in config.NUMERIC_IDX else row[c]
                              for i, c in enumerate(config.FEATURES)])
    return ids, instances


def post_batch(url, instances):
    """One attempt. Returns the churn probability of each row, or raises RuntimeError."""
    request = urllib.request.Request(url, data=json.dumps({"instances": instances}).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"answered {e.code}") from e
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"failed: {getattr(e, 'reason', e)}") from e
    if status != 200:
        raise RuntimeError(f"answered {status}")
    try:
        predictions = json.loads(body).get("predictions")
    except (ValueError, AttributeError) as e:
        raise RuntimeError("the answer is not JSON with 'predictions'") from e
    if not isinstance(predictions, list) or len(predictions) != len(instances):
        raise RuntimeError(f"expected {len(instances)} predictions, got "
                           f"{len(predictions) if isinstance(predictions, list) else type(predictions).__name__}")
    probabilities = []
    for pair in predictions:
        if (not isinstance(pair, list) or len(pair) != 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in pair)):
            raise RuntimeError(f"a prediction is not a pair of probabilities in [0, 1]: {str(pair)[:80]}")
        probabilities.append(float(pair[1]))
    return probabilities


def send_with_retries(args, counters, instances, label):
    """The first attempt plus up to `retries` more. Every attempt counts in the metric."""
    last = None
    for attempt in range(args.retries + 1):
        started = time.monotonic()
        try:
            probabilities = post_batch(args.url, instances)
        except RuntimeError as e:
            counters.add("error")
            last = e
            print(f"{label} attempt {attempt + 1}/{args.retries + 1}: {e}", flush=True)
            if attempt < args.retries:
                time.sleep(args.retry_pause)
            continue
        counters.add("ok")
        return probabilities, time.monotonic() - started
    raise RuntimeError(f"{label} failed after {args.retries + 1} attempts: {last}")


def write_submission(path, ids, probabilities):
    """id,Exited through a temporary file and a rename, so a reader never sees half a file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([config.ID_COL, config.TARGET])
        writer.writerows(zip(ids, probabilities))
    os.replace(tmp, path)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def run(args, counters):
    """Run every pass. Raises RuntimeError if a batch fails after its retries."""
    for stale in (args.output, f"{args.output}.tmp"):
        if os.path.exists(stale):
            os.remove(stale)
            print(f"removed a stale {stale}", flush=True)
    ids, instances = read_rows(args.data)
    batches = [(i, instances[i:i + args.batch_size]) for i in range(0, len(instances), args.batch_size)]
    interval = 1.0 / args.rate
    print(f"{len(instances)} rows, {len(batches)} requests per pass, {args.passes} passes, "
          f"{args.rate} requests/s, endpoint {args.url}", flush=True)

    next_slot = time.monotonic()
    for number in range(1, args.passes + 1):
        probabilities, latencies, started = [], [], time.monotonic()
        for start, batch in batches:
            time.sleep(max(0.0, next_slot - time.monotonic()))
            answer, latency = send_with_retries(args, counters, batch, f"pass {number} rows {start}-{start + len(batch) - 1}")
            next_slot = max(next_slot + interval, time.monotonic())
            latencies.append(latency)
            if number == 1:
                probabilities.extend(answer)
        elapsed = time.monotonic() - started
        print(f"pass {number}: {len(batches)} requests in {elapsed:.1f} s = {len(batches) / elapsed:.2f} requests/s, "
              f"batch latency p50 {percentile(latencies, 0.5):.3f} s, p95 {percentile(latencies, 0.95):.3f} s, "
              f"max {max(latencies):.3f} s", flush=True)
        if number == 1:
            write_submission(args.output, ids, probabilities)
            print(f"wrote {args.output}: {len(probabilities)} rows", flush=True)
    print(f"done: ok {counters.values['ok']}, error {counters.values['error']}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--data", default=f"{config.RAW_DIR_PVC}/test.csv")
    p.add_argument("--output", default=config.SUBMISSION_PATH)
    p.add_argument("--batch-size", type=int, default=config.LOAD_BATCH_SIZE)
    p.add_argument("--rate", type=float, default=config.LOAD_RATE_RPS, help="requests per second")
    p.add_argument("--passes", type=int, default=config.LOAD_PASSES)
    p.add_argument("--retries", type=int, default=config.LOAD_RETRIES)
    p.add_argument("--retry-pause", type=float, default=config.LOAD_RETRY_PAUSE_S)
    p.add_argument("--metrics-port", type=int, default=config.LOAD_METRICS_PORT)
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.passes < 1 or args.rate <= 0 or args.retries < 0:
        p.error("--batch-size and --passes must be at least 1, --rate above 0, --retries not negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    counters = Counters()
    start_metrics_server(counters, args.metrics_port)
    try:
        run(args, counters)
    except RuntimeError as e:
        print(f"FAILED: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
