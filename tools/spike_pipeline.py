from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_IDX, NUMERIC_IDX


def build_pipeline():
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_IDX),
            ("num", StandardScaler(), NUMERIC_IDX),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
