# Decisions

Single source of the frozen contracts. Changing a row needs a new line in the log below **first**, then the code and docs.

## Frozen contracts

| Contract | Value |
|---|---|
| Target | `Exited` (0/1), output is a probability |
| Dropped columns | `id`, `CustomerId`, `Surname` (`id` kept aside only for the submission) |
| `FEATURES` order | `CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary` |
| Split | Stratified 80/20 train/hold-out, `SEED = 42` |
| Main metric | ROC-AUC (plus PR-AUC, recall, F1, confusion matrix) |
| Gate | Hold-out ROC-AUC > Dummy baseline + 0.05 **and** ≥ current champion's hold-out AUC |
| Registry | Model `churn`, alias `@champion` |
| scikit-learn | `1.5.2`, pinned to the KServe 0.21 sklearnserver (`kserve/sklearnserver:v0.21.0`, Python 3.11.16) |

## Log

- **2026-10-30 (SPEC 01):** contracts above written from README §2 and §5. `SEED = 42` chosen because the README asked for a fixed seed without a value.
- **2026-10-30 (SPEC 01, E0.2):** scikit-learn pinned to `1.5.2` and Python to `3.11`, read from the sklearnserver image (see `docs/runbook.md`). The server also ships numpy 2.4.4, which SPEC 02 must verify against a model pickled by `churn-base`.
