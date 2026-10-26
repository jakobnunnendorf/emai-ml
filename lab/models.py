"""The four model kinds as sklearn Pipelines.

linear, quadratic, logistic, quadratic_logistic. All preprocessing (imputing,
degree-2 expansion, scaling, one-hot) is inside the pipeline, so it is fitted
on the training sample only.

Notes:
- Only numeric columns get the degree-2 expansion. Squares and products of
  one-hot dummies add many useless columns.
- No regularization. With p > n, LinearRegression returns the minimum-norm
  least-squares solution (training MSE 0). LogisticRegression(C=inf) on
  separable samples does not converge and gets very over-confident.
"""

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

REGRESSION_KINDS = ["linear", "quadratic"]
CLASSIFICATION_KINDS = ["logistic", "quadratic_logistic"]


def make_preprocessor(quadratic: bool) -> ColumnTransformer:
    """Numeric columns: impute, (square), scale. Other columns: impute, one-hot."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if quadratic:
        numeric_steps.append(("square", PolynomialFeatures(degree=2, include_bias=False)))
    numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        # a small sample may miss some categories
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
            ("cat", Pipeline(categorical_steps), make_column_selector(dtype_exclude=np.number)),
        ],
        sparse_threshold=0.0,
    )


def make_model(kind: str) -> Pipeline:
    quadratic = kind.startswith("quadratic")

    if kind in REGRESSION_KINDS:
        estimator = LinearRegression()
    elif kind in CLASSIFICATION_KINDS:
        # C=inf -> no penalty; multinomial for > 2 classes
        estimator = LogisticRegression(C=np.inf, max_iter=5000)
    else:
        raise ValueError(f"unknown model kind: {kind!r}")

    return Pipeline([("prep", make_preprocessor(quadratic)), ("model", estimator)])


def make_baseline(task: str):
    """Predict the training mean (regression) or class frequencies (classification)."""
    if task == "regression":
        return DummyRegressor(strategy="mean")
    return DummyClassifier(strategy="prior")
