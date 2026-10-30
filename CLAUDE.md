# Bank Customer Churn — MLOps on Kubeflow

Portfolio project: raw CSV to a monitored, served model on minikube (KFP, MLflow, KServe, Prometheus/Grafana).

## Read first
- [README.md](README.md): goal, scope, architecture, design decisions, phases.
- [BACKLOG.md](BACKLOG.md): epics and subtasks.
- [DECISIONS.md](DECISIONS.md): frozen contracts. Changing one needs a new line there first.
- `specs/`: one spec per unit of work, driven by `/spec` and `/spec-impl`.

## Rules
- Docs and code are in English, even when the conversation is in Spanish.
- Never commit unless the user explicitly asks.
- Data in `data/` is for learning only: never commit or publish it.
- Docker commands are run by the user.
