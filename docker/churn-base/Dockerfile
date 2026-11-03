# Base image for every pipeline step and the load generator.
# Python and scikit-learn match kserve/sklearnserver:v0.21.0 (see DECISIONS.md).
# requirements.txt is generated, never edited by hand:
#   uv export --no-dev --no-hashes --no-emit-package nvidia-nccl-cu12 --format requirements-txt > docker/churn-base/requirements.txt
FROM python:3.11-slim

WORKDIR /app

# libgomp1: OpenMP runtime that LightGBM needs and python:3.11-slim does not carry.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --no-deps: requirements.txt is a complete, pinned export. Without it pip would also install
# nvidia-nccl-cu12 (352 MB of GPU code), which xgboost declares and which was left out on purpose.
RUN pip install --no-cache-dir --no-deps -r requirements.txt

ENV PYTHONUNBUFFERED=1
