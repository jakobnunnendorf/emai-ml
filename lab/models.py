"""Model factory: every model we use in the lab, built as an sklearn Pipeline.

Why a Pipeline?
    All preprocessing (imputing missing values, squaring features, scaling,
    one-hot encoding) lives *inside* the model. When we call
    ``model.fit(X_sample, y_sample)`` every preprocessing step is fitted on
    that sample only, so nothing leaks from the validation set (or from the
    rest of the training data) into the model. The data loaders therefore
    hand us raw-but-clean DataFrames and never scale or encode anything.

The four model kinds
    ``linear``              LinearRegression on the preprocessed features
    ``quadratic``           same, but numeric features are first expanded to
                            all degree-2 terms (x_i, x_i**2, x_i * x_j)
    ``logistic``            LogisticRegression (multinomial for > 2 classes)
    ``quadratic_logistic``  LogisticRegression on the degree-2 expansion

Design decisions (keep these in mind when explaining the plots)
    1. Only NUMERIC columns are squared. Categorical columns are one-hot
       encoded and passed through unchanged. Squaring a 0/1 dummy gives the
       same dummy again, and products of two dummies are almost always zero,
       so expanding them would add thousands of mostly useless columns
       (automobile: 67 columns -> ~2,300) without adding information.
       Automobile has 17 numeric columns -> 17 + 17 + 136 = 170 quadratic
       terms, plus 50 one-hot columns = 220 features (linear model: 67).

    2. No regularization, because the assignment asks for plain linear /
       logistic regression.
       - LinearRegression solves least squares with ``scipy.linalg.lstsq``.
         When there are more features p than samples n the problem has
         infinitely many perfect fits; lstsq returns the one with the
         smallest coefficient norm ("minimum-norm solution"). Consequences:
           * training MSE is ~0 whenever n <= p (the model can interpolate),
           * validation MSE is typically largest around n ~ p and drops
             again on both sides (a "double descent" shaped curve).
       - LogisticRegression(C=inf), i.e. no penalty: on tiny samples the classes are
         often perfectly separable, so the optimal weights are infinite. The
         solver stops at ``max_iter`` with a ConvergenceWarning, the model is
         extremely over-confident and the validation log-loss is huge. That
         is expected behaviour and part of what the learning curves show.

    3. Features are standardized (mean 0, std 1) after the optional squaring.
       This does not change what plain linear regression can fit, but it
       makes the optimization for logistic regression much better behaved.
"""

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

# Which kinds are regression and which are classification models.
REGRESSION_KINDS = ["linear", "quadratic"]
CLASSIFICATION_KINDS = ["logistic", "quadratic_logistic"]


def make_preprocessor(quadratic: bool) -> ColumnTransformer:
    """Turn a raw DataFrame into a numeric feature matrix.

    Columns are picked by dtype, so the same code works for every dataset:
    numeric dtypes go to the numeric branch, everything else (strings,
    categories) to the categorical branch.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if quadratic:
        # degree=2 adds squares and pairwise products. include_bias=False
        # because the regression model already fits an intercept.
        numeric_steps.append(("square", PolynomialFeatures(degree=2, include_bias=False)))
    numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a small training sample may not contain
        # every category; unseen categories in validation become all-zeros.
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
            ("cat", Pipeline(categorical_steps), make_column_selector(dtype_exclude=np.number)),
        ],
        # Return a dense numpy array; our feature counts are small.
        sparse_threshold=0.0,
    )


def make_model(kind: str) -> Pipeline:
    """Build a fresh, unfitted model of the given kind (see module docstring)."""
    quadratic = kind.startswith("quadratic")

    if kind in REGRESSION_KINDS:
        estimator = LinearRegression()
    elif kind in CLASSIFICATION_KINDS:
        # C is the inverse regularization strength (sklearn's default C=1
        # adds an L2 penalty). C=inf -> no penalty -> plain logistic regression.
        # With > 2 classes sklearn fits a multinomial (softmax) model.
        estimator = LogisticRegression(C=np.inf, max_iter=5000)
    else:
        raise ValueError(f"unknown model kind: {kind!r}")

    return Pipeline([("prep", make_preprocessor(quadratic)), ("model", estimator)])


def make_baseline(task: str):
    """The simplest possible model, used as a reference line in the plots.

    regression:     always predict the mean of the training targets
    classification: always predict the class frequencies of the training set
    A useful model should end up below this line.
    """
    if task == "regression":
        return DummyRegressor(strategy="mean")
    return DummyClassifier(strategy="prior")
