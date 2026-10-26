"""Learning curves (steps 1, 2, 4) and cross-validation (step 3).

Scores are losses (lower is better): MSE for regression, log-loss for
classification.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from lab.models import make_baseline, make_model


def score(model, X, y, task: str, classes=None) -> float:
    """MSE or log-loss of a fitted model. `classes` = all classes of the dataset."""
    if task == "regression":
        return mean_squared_error(y, model.predict(X))

    if len(model.classes_) != len(classes):
        raise ValueError(
            f"model was trained on classes {list(model.classes_)} but the dataset "
            f"has {list(classes)}; every training sample must contain all classes"
        )
    return log_loss(y, model.predict_proba(X), labels=classes)


def run_learning_curve(X_train, y_train, X_val, y_val, kinds, samples, task, verbose=True):
    """Fit every model kind on every sample from the sampler and score it.

    samples: (size, repeat, idx) tuples, idx = positions in X_train.
    Returns one row per (model, size, repeat) with train_score, val_score and
    fit_seconds.
    """
    classes = np.unique(y_train) if task == "classification" else None
    rows = []
    last_size = None

    # unregularized logistic regression on separable samples never converges
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        for size, repeat, idx in samples:
            if verbose and size != last_size:
                print(f"size {size:>8}", flush=True)
                last_size = size

            X_sample, y_sample = X_train.iloc[idx], y_train.iloc[idx]
            for kind in kinds:
                start = time.perf_counter()
                model = make_model(kind).fit(X_sample, y_sample)
                rows.append({
                    "model": kind,
                    "size": size,
                    "repeat": repeat,
                    "train_score": score(model, X_sample, y_sample, task, classes),
                    "val_score": score(model, X_val, y_val, task, classes),
                    "fit_seconds": time.perf_counter() - start,
                })

    return pd.DataFrame(rows)


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def summarize(curve: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, median and quartiles over the repeats, per model and size.
    The plots use median and quartiles (single bad repeats dominate the mean)."""
    return (
        curve.groupby(["model", "size"])[["train_score", "val_score"]]
        .agg(["mean", "std", "median", q25, q75])
        .reset_index()
    )


def baseline_score(X_train, y_train, X_val, y_val, task) -> float:
    """Validation score of the trivial baseline fitted on the whole training part."""
    classes = np.unique(y_train) if task == "classification" else None
    model = make_baseline(task).fit(X_train, y_train)
    return score(model, X_val, y_val, task, classes)


def cross_validate_models(X_train, y_train, kinds, task, k=5, folds=None, groups=None, max_rows=None, seed=42):
    """k-fold CV of each model kind on the training part.

    folds:    sklearn splitter (e.g. GroupKFold); default shuffled KFold /
              StratifiedKFold
    groups:   group per training row, for group splitters
    max_rows: run CV on a random subset (big datasets)

    Returns columns model, cv_mean, cv_std, n_rows.
    """
    if max_rows is not None and len(X_train) > max_rows:
        X_train = X_train.sample(n=max_rows, random_state=seed)
        y_train = y_train.loc[X_train.index]
        if groups is not None:
            groups = groups.loc[X_train.index]

    if task == "regression":
        default_folds = KFold(n_splits=k, shuffle=True, random_state=seed)
        scoring = "neg_mean_squared_error"
    else:
        default_folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        scoring = "neg_log_loss"
    if folds is None:
        folds = default_folds

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for kind in kinds:
            # sklearn scorers are negated losses
            fold_scores = -cross_val_score(make_model(kind), X_train, y_train, groups=groups,
                                           cv=folds, scoring=scoring)
            rows.append({
                "model": kind,
                "cv_mean": fold_scores.mean(),
                "cv_std": fold_scores.std(),
                "n_rows": len(X_train),
            })
    return pd.DataFrame(rows)
