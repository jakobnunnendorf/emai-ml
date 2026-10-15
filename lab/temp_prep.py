"""TEMPORARY stand-in for Jakob's data-prep modules (lab/split.py, lab/sampler.py).

Same function names and signatures as agreed in HANDOFF_DATA_PREP.md, so the
notebooks can already run. Once Jakob's versions exist, change the imports in
the notebooks and delete this file.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def train_val_split(X, y, task, seed=42):
    """Set aside 20% of the data for validation (stratified for classification)."""
    stratify = y if task == "classification" else None
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=stratify)


def linear_sizes(start, stop, step):
    """Evenly spaced sizes, e.g. linear_sizes(10, 160, 10) -> 10, 20, ..., 160."""
    return list(range(start, stop + 1, step))


def log_sizes(lo, hi, num):
    """Roughly log-spaced integer sizes from lo to hi (duplicates removed)."""
    return sorted({int(round(s)) for s in np.geomspace(lo, hi, num)})


def sample_training_sets(y_train, sizes, n_repeats, task, seed=0):
    """Yield (size, repeat, idx): idx are positional indices into the training part.

    Sampling is without replacement. For classification the sample is
    stratified and must contain every class (otherwise log-loss is undefined).
    Each (size, repeat) gets its own reproducible random seed.
    """
    n = len(y_train)
    positions = np.arange(n)
    n_classes = y_train.nunique()

    for size in sizes:
        for repeat in range(n_repeats):
            rs = int(np.random.SeedSequence([seed, size, repeat]).generate_state(1)[0])
            if size >= n:
                # The whole training part; every repeat is the same set.
                yield size, repeat, positions
                continue

            if task == "classification":
                if size < n_classes:
                    raise ValueError(f"size {size} is smaller than the number of classes ({n_classes})")
                idx, _ = train_test_split(positions, train_size=size, random_state=rs, stratify=y_train)
                if y_train.iloc[idx].nunique() < n_classes:
                    raise ValueError(f"sample of size {size} is missing a class; use larger sizes")
            else:
                idx = np.random.default_rng(rs).choice(n, size=size, replace=False)
            yield size, repeat, idx
