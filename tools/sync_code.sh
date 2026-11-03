#!/usr/bin/env bash
# Refresh the ConfigMap `churn-code` with the shared code the pipeline steps import.
# Steps mount it at /code (PYTHONPATH=/code), so a code change needs no image rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."

kubectl create configmap churn-code -n kubeflow \
  --from-file=config.py=components/config.py \
  --from-file=preprocessing.py=components/preprocessing.py \
  --dry-run=client -o yaml | kubectl apply -f -
