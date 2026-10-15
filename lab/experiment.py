"""The experiments of the lab: learning curves (steps 1, 2, 4) and
cross-validation (step 3).

Everything returns plain pandas DataFrames, so results can be saved to CSV,
inspected in a notebook, and plotted with ``lab.plotting``.

Scores are always "lower is better":
    regression      -> mean squared error (MSE)
    classification  -> logistic loss (log-loss / cross-entropy)
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
    """Score a fitted model on (X, y). Lower is better.

    For classification ``classes`` must list ALL classes of the dataset, so
    log-loss is computed over the same set of classes for every sample.
    """
    if task == "regression":
        return mean_squared_error(y, model.predict(X))

    # The model can only output probabilities for classes it saw during
    # training. The sampler guarantees every class is present; check anyway
    # so a bad sample fails loudly instead of giving a wrong number.
    if len(model.classes_) != len(classes):
        raise ValueError(
            f"model was trained on classes {list(model.classes_)} but the dataset "
            f"has {list(classes)}; every training sample must contain all classes"
        )
    return log_loss(y, model.predict_proba(X), labels=classes)


def run_learning_curve(X_train, y_train, X_val, y_val, kinds, samples, task, verbose=True):
    """Fit every model kind on every sampled training set and score it.

    Parameters
    ----------
    X_train, y_train : the 80% training part (DataFrame / Series)
    X_val, y_val     : the 20% validation part, never used for fitting
    kinds            : list of model kinds, e.g. ["linear", "quadratic"]
    samples          : iterable of (size, repeat, idx) from the sampler, where
                       idx are positional indices into X_train
    task             : "regression" or "classification"

    Returns
    -------
    DataFrame with one row per (model, size, repeat):
        model, size, repeat, train_score, val_score, fit_seconds
    train_score is measured on the sample the model was fitted on.
    """
    classes = np.unique(y_train) if task == "classification" else None
    rows = []
    last_size = None

    # Unregularized logistic regression on tiny (separable) samples never
    # "converges" because the best weights are infinitely large. sklearn warns
    # about it on every fit; that is expected here (see lab/models.py), so we
    # silence just that warning to keep the output readable.
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
    """Statistics over the repeats, per model and size.

    Besides mean and std we also report the median and the 25% / 75%
    quantiles. With tiny samples a single unlucky repeat can have a loss
    1000x larger than the others; such outliers dominate mean and std, while
    median and quantiles describe the typical behaviour. The plots use those.
    """
    return (
        curve.groupby(["model", "size"])[["train_score", "val_score"]]
        .agg(["mean", "std", "median", q25, q75])
        .reset_index()
    )


def baseline_score(X_train, y_train, X_val, y_val, task) -> float:
    """Validation score of the trivial baseline (see lab.models.make_baseline),
    fitted on the full training part. Used as a horizontal line in the plots."""
    classes = np.unique(y_train) if task == "classification" else None
    model = make_baseline(task).fit(X_train, y_train)
    return score(model, X_val, y_val, task, classes)


def cross_validate_models(X_train, y_train, kinds, task, k=5, folds=None, groups=None, max_rows=None, seed=42):
    """Step 3: k-fold cross-validation of each model kind on the training part.

    The validation set is NOT used here. CV splits the training part itself
    into k folds, trains on k-1 of them and scores on the remaining one, k
    times. The mean over the folds estimates how well the model generalizes.

    folds:    optional sklearn splitter, so CV can follow the same logic as the
              dataset's train/validation split (e.g. GroupKFold to hold out whole
              months). Default: shuffled KFold (regression) or StratifiedKFold
              (classification) with k folds.
    groups:   group label per training row (Series with the same index as
              X_train), needed by group splitters such as GroupKFold.
    max_rows: optionally run CV on a random subset of the training part
              (useful for the big datasets, where CV on all rows is slow).

    Returns a DataFrame with columns: model, cv_mean, cv_std, n_rows.
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
        # Stratified: every fold keeps the class proportions.
        default_folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        scoring = "neg_log_loss"
    if folds is None:
        folds = default_folds

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for kind in kinds:
            # sklearn scorers follow "higher is better", so MSE / log-loss come
            # back negated. Flip the sign to get the usual loss values.
            fold_scores = -cross_val_score(make_model(kind), X_train, y_train, groups=groups,
                                           cv=folds, scoring=scoring)
            rows.append({
                "model": kind,
                "cv_mean": fold_scores.mean(),
                "cv_std": fold_scores.std(),
                "n_rows": len(X_train),
            })
    return pd.DataFrame(rows)
