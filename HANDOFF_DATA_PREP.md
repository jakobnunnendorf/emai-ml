# Handoff: what the pipeline expects from data prep

Hi Jakob, this is the interface between data prep and the shared pipeline
(`lab/models.py`, `lab/experiment.py`, `lab/plotting.py`). If you stick to these
function names and return types, the notebooks only need their imports changed.

## Who does what
| Jakob | Tomi |
|---|---|
| `lab/data_wine.py`: `load_wine()`, `split_wine()` | `lab/data_automobile.py`: `load_automobile()`, `split_automobile()` |
| `lab/data_power.py`: `load_power()`, `split_power()` | `lab/data_taxi.py`: `load_taxi()`, `split_taxi()` |
| `lab/sampler.py`: `sample_training_sets()`, `linear_sizes()`, `log_sizes()` | models + preprocessing, learning-curve loop, metrics, CV, plots |

**Important: do NOT scale, one-hot encode or impute in data prep.** The pipeline does
that inside the model, so it is fitted on each training sample only (no leakage).

## 1. Loaders: `load_<name>()` in `lab/data_<name>.py`
Returns `X, y`, plus any columns that are needed **only for splitting** (e.g. a
timestamp) as extra return values. Example: `load_taxi()` returns `X, y, time`.
- `X`: one row per data point, one column per **feature**.
  - numeric features have a **numeric dtype** (int / float),
  - categorical features have dtype `object`, `str` or `category`.
  The pipeline picks columns **by dtype**: numeric -> imputed, (squared,) scaled;
  everything else -> one-hot. So an integer that is really a label (e.g. hour of day)
  should be converted with `.astype("category")`.
- `y`: target Series, **no missing values** (drop those rows).
- Missing values in `X` are fine as `NaN` (the pipeline imputes them).
- No leakage columns, no ID columns, no split-only columns (timestamp etc.) in `X`.
- Read paths via `from lab import DATA_DIR` (respects the `LAB_DATA_DIR` env var).

## 2. Split: `split_<name>(...)` next to the loader
**Every dataset gets its own small split function**, because the right split depends on
the data. Write it with sklearn's building blocks (`train_test_split(stratify=...)`,
`GroupShuffleSplit`, or a date cutoff); it is usually 2-5 lines. Requirements:
- returns `X_train, X_val, y_train, y_val` (pandas objects, like `train_test_split`),
- about 20% validation, deterministic (fixed seed or fixed cutoff),
- for classification: every class present in both parts,
- the docstring says **why** this split (that text goes into the report).

Rule of thumb: decide what one *independent unit* is (a car, a wine, a day, a week) and
never put near-copies of the same unit on both sides of the split.

Examples in the repo:
- `split_automobile`: random, stratified on price quintiles (skewed target, tiny set).
- `split_taxi`: temporal, last ~20% of days are validation (realistic + same-day trips
  stay together). Its CV uses whole months as folds, see section 4.

## 3. Sampler: `lab/sampler.py` (generic, same for all datasets)
```python
def sample_training_sets(y_train, sizes, n_repeats, task, seed=0):
    """Yield (size, repeat, idx) for every size and repeat."""

def linear_sizes(start, stop, step) -> list[int]   # 10, 20, ..., stop
def log_sizes(lo, hi, num) -> list[int]            # ~log-spaced unique ints lo..hi
```
- Samples are drawn **randomly from the training part**, whatever the split was:
  the learning curve is about the *number* of training points.
- `idx` = **positional** indices into `X_train` / `y_train` (we use `.iloc[idx]`).
- Sampling **without replacement**, each `(size, repeat)` with its own reproducible
  seed (e.g. `np.random.SeedSequence([seed, size, repeat])`).
- Classification: stratified, and **every class must appear in every sample**,
  otherwise log-loss cannot be computed. Raise an error for sizes < number of classes.
- If `size >= len(y_train)`: return all rows (every repeat is then identical).

`lab/temp_prep.py` is a working version with exactly these signatures; copy it into
`lab/sampler.py` and improve it, or write your own. Then delete `temp_prep.py` and switch
the imports in the notebooks (marked `# TODO`).

## 4. Cross-validation should follow the split
`cross_validate_models(..., folds=None, groups=None)` uses shuffled (Stratified)KFold by
default. If your split is grouped or temporal, pass the matching sklearn splitter, e.g.
```python
from sklearn.model_selection import GroupKFold
cross_validate_models(X_train, y_train, kinds, task,
                      folds=GroupKFold(n_splits=5), groups=week_of_each_train_row)
```
`groups` is a Series with the same index as `X_train`.

## 5. Dataset notes
- **wine:** class = column 1 (values 1-3), 13 numeric features, no missing values.
  Independent samples -> random split, stratified on the class. Default CV is fine.
- **household power** (`;`-separated, `?` = missing, one row per minute):
  - target `Global_active_power`; drop rows where it is missing.
  - drop `Global_intensity` (it is ~ power / voltage, i.e. leaks the target).
  - `Global_reactive_power`, `Voltage`, `Sub_metering_1..3` are numeric features.
  - `Date` / `Time` -> e.g. `hour`, `weekday`, `month` as features; keep the timestamp
    separately for splitting.
  - **Split:** neighbouring minutes are almost identical, so a random minute-level split
    leaks heavily and looks far too good. Keep whole blocks together: e.g. the last 20%
    in time, or random whole weeks (`GroupShuffleSplit` with week as group); CV with
    `GroupKFold` by week/month. Comparing random vs block split would be a nice
    leakage example for the report.
  - 2M rows: a `max_rows` option for quick local runs is handy.

## 6. How a notebook uses it
```python
from lab.data_wine import load_wine, split_wine
from lab.sampler import sample_training_sets, linear_sizes
from lab.experiment import run_learning_curve, baseline_score, cross_validate_models
from lab.plotting import plot_learning_curve

X, y = load_wine()
X_train, X_val, y_train, y_val = split_wine(X, y)
sizes = linear_sizes(10, len(X_train), 10)
samples = sample_training_sets(y_train, sizes, n_repeats=10, task="classification")
curve = run_learning_curve(X_train, y_train, X_val, y_val,
                           ["logistic", "quadratic_logistic"], samples, "classification")
baseline = baseline_score(X_train, y_train, X_val, y_val, "classification")
plot_learning_curve(curve, "log-loss", "Wine", baseline=baseline, log_y=True,
                    path="figures/wine_learning_curves.png")
cv = cross_validate_models(X_train, y_train, ["logistic", "quadratic_logistic"], "classification")
```
`notebooks/automobile.ipynb` and `notebooks/taxi.ipynb` are complete examples; copying one
of them is the fastest way to make `wine.ipynb` / `power.ipynb`.
