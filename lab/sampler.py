"""Random training samples of given sizes, drawn from the training part.

Without replacement, each (size, repeat) has its own seed. For classification
the samples are stratified and contain every class (else log-loss is
undefined). Samples of different sizes are independent, not nested.
"""

import numpy as np


def linear_sizes(start, stop, step):
    """linear_sizes(10, 160, 10) -> 10, 20, ..., 160"""
    return list(range(start, stop + 1, step))


def log_sizes(lo, hi, num):
    """About `num` log-spaced integer sizes from lo to hi (duplicates removed)."""
    return sorted({int(round(s)) for s in np.geomspace(lo, hi, num)})


def _seed(seed, size, repeat):
    return int(np.random.SeedSequence([seed, size, repeat]).generate_state(1)[0])


def sample_training_sets(y_train, sizes, n_repeats, task, seed=0):
    """Yield (size, repeat, idx), idx = positions in the training part.

    A size >= len(y_train) gives all rows (every repeat is the same).
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
    """Rows per class: proportional (largest remainder), at least 1 per class."""
    exact = size * class_sizes / class_sizes.sum()
    counts = np.maximum(np.floor(exact).astype(int), 1)
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
    # not train_test_split(stratify=...): it also needs every class in the
    # left-out part, which fails for sizes close to len(y)
    classes, inverse, class_sizes = np.unique(y, return_inverse=True, return_counts=True)
    counts = _class_counts(class_sizes, size)
    parts = [rng.choice(np.flatnonzero(inverse == c), size=k, replace=False) for c, k in enumerate(counts)]
    return np.concatenate(parts)
