"""Shared preprocessing for the churn model. Columns are addressed by position, so
the same transformer works on a DataFrame or a NumPy array in FEATURES order."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_IDX, NUMERIC_IDX


def build_preprocessor():
    """Return an UNFITTED transformer: one-hot for the categorical columns and
    scaling for the numeric ones. It is fitted inside each CV fold (Tune) and on
    train + hold-out (Register), never here."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_IDX),
        ("num", StandardScaler(), NUMERIC_IDX),
    ])
