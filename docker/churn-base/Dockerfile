# Base image for every pipeline step and the load generator.
# Python and scikit-learn match kserve/sklearnserver:v0.21.0 (see DECISIONS.md).
# requirements.txt is generated, never edited by hand:
#   uv export --no-dev --no-hashes --format requirements-txt > docker/churn-base/requirements.txt
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
