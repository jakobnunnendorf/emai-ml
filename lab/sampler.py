"""Training-set sampler for the learning curves (generic, same for all datasets).

A learning curve plots the error against the NUMBER of training points. For
every size n we draw several random samples of n rows from the training part,
fit a model on each, and look at the spread over these repeats.

Rules (see HANDOFF_DATA_PREP.md, section 3):
- sizes are absolute numbers of rows (10, 20, ...), not fractions,
- samples are drawn at random from the training part, whatever the
  train/validation split was (that is a dataset decision, see lab/data_*.py),
- without replacement, and every (size, repeat) has its own reproducible seed,
  so a single sample can be re-created without running all the others,
- classification: stratified, and every class must be in every sample,
  otherwise the model cannot output a probability for the missing class and
  log-loss is undefined.

The samples are independent between sizes (the sample of size 20 does not
contain the one of size 10). Nested samples would make the curves smoother,
but independent ones show the real variability of "a random training set of
this size", which is what the shaded bands in the plots are meant to show.
"""

import numpy as np


def linear_sizes(start, stop, step):
    """Evenly spaced sizes, e.g. linear_sizes(10, 160, 10) -> 10, 20, ..., 160.
    For the small datasets, where every step of 10 rows matters."""
    return list(range(start, stop + 1, step))


def log_sizes(lo, hi, num):
    """Roughly log-spaced integer sizes from lo to hi, e.g. 10, 20, 50, ..., 1M
    (duplicates removed, so the result can be shorter than num).
    For the big datasets: the interesting part of the curve is at small n, and
    a log axis shows 10 -> 100 and 100k -> 1M with the same resolution."""
    return sorted({int(round(s)) for s in np.geomspace(lo, hi, num)})


def _seed(seed, size, repeat):
    """Reproducible random seed for one (size, repeat) pair."""
    return int(np.random.SeedSequence([seed, size, repeat]).generate_state(1)[0])


def sample_training_sets(y_train, sizes, n_repeats, task, seed=0):
    """Yield (size, repeat, idx) for every size and repeat.

    idx are POSITIONAL indices into the training part (use X_train.iloc[idx]).

    y_train   training targets (Series); for classification used to stratify
    sizes     list of sample sizes, e.g. from linear_sizes / log_sizes
    n_repeats number of different random samples per size
    task      "regression" or "classification"
    seed      base seed; change it to get a completely different set of samples

    If a size is >= the training part, all rows are used (every repeat is then
    the same set, so the spread at that size is 0).
    """
    n = len(y_train)
    positions = np.arange(n)

    if task == "classification":
        n_classes = y_train.nunique()
        too_small = [size for size in sizes if size < n_classes]
        if too_small:
            raise ValueError(f"sizes {too_small} are smaller than the number of classes ({n_classes})")

    for size in sizes:
        for repeat in range(n_repeats):
            if size >= n:
                yield size, repeat, positions
                continue

            rng = np.random.default_rng(_seed(seed, size, repeat))
            if task == "classification":
                idx = _stratified_sample(y_train.to_numpy(), size, rng)
            else:
                idx = rng.choice(n, size=size, replace=False)
            yield size, repeat, np.sort(idx)


def _class_counts(class_sizes, size):
    """How many rows to draw from each class: proportional to the class sizes
    (largest-remainder rounding), but at least 1 per class."""
    exact = size * class_sizes / class_sizes.sum()
    counts = np.maximum(np.floor(exact).astype(int), 1)
    # Hand out the rows still missing (or take back extra ones, if the
    # "at least 1" rule overshot) by the size of the rounding remainder.
    order = np.argsort(-(exact - np.floor(exact)))
    i = 0
    while counts.sum() != size:
        c = order[i % len(order)]
        if counts.sum() < size and counts[c] < class_sizes[c]:
            counts[c] += 1
        elif counts.sum() > size and counts[c] > 1:
            counts[c] -= 1
        i += 1
    return counts


def _stratified_sample(y, size, rng):
    """Positions of a random sample of `size` rows with the class proportions
    of y and every class present.

    (sklearn's train_test_split(stratify=...) is not used: it also requires
    the LEFT-OUT part to contain every class, which fails for sizes close to
    the whole training part, e.g. 140 of 142 wines.)
    """
    classes, inverse, class_sizes = np.unique(y, return_inverse=True, return_counts=True)
    counts = _class_counts(class_sizes, size)
    parts = [rng.choice(np.flatnonzero(inverse == c), size=k, replace=False) for c, k in enumerate(counts)]
    return np.concatenate(parts)
