"""Query step of the serving spike (E0.7).

Reads spike-reference.json, POSTs its raw instances to the KServe v1 endpoint and
prints the HTTP status, the body, the shape of "predictions" and the maximum
absolute difference against the expected predict_proba. A 4xx or 5xx is a valid
finding.

Needs: kubectl port-forward -n kubeflow svc/churn-spike-predictor 8080:80
Pass another local port as the first argument if 8080 is taken.
"""
import json
import sys
import urllib.error
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8080"
URL = f"http://localhost:{PORT}/v1/models/churn-spike:predict"
REFERENCE = "spike-reference.json"


def main():
    with open(REFERENCE) as f:
        reference = json.load(f)

    request = urllib.request.Request(
        URL,
        data=json.dumps({"instances": reference["instances"]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, body = response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read().decode()

    print(f"status: {status}")
    print(f"body: {body}")

    try:
        predictions = json.loads(body)["predictions"]
    except (ValueError, KeyError, TypeError):
        print("no 'predictions' in the body")
        return

    shape = (len(predictions), len(predictions[0]) if isinstance(predictions[0], list) else None)
    print(f"predictions shape: {shape}")
    if shape[1] == len(reference["expected"][0]):
        diff = max(abs(p - e) for row, exp in zip(predictions, reference["expected"]) for p, e in zip(row, exp))
        print(f"max abs diff vs expected: {diff:.3e}")
    else:
        print("shape differs from expected, no difference computed")


if __name__ == "__main__":
    main()
