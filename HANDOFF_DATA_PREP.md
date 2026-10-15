# Handoff: what the pipeline expects from data prep

Hi Jakob, this is the interface between your data prep and the shared pipeline
(`lab/models.py`, `lab/experiment.py`, `lab/plotting.py`). If you stick to these
function names and return types, the notebooks only need their imports changed.

`lab/temp_prep.py` is a working minimal version of the split and sampler with exactly
these signatures. Feel free to copy it into `lab/split.py` / `lab/sampler.py` and
improve it, or write your own. Once yours exist, delete `temp_prep.py` and switch the
import in the notebooks (marked `# TODO`).

## Who does what
| Data prep (Jakob) | Pipeline (Tomi) |
|---|---|
| `load_wine()`, `load_power()` | `load_automobile()`, `load_taxi()` |
| `train_val_split()` | models + preprocessing (`make_model`) |
| `sample_training_sets()`, `linear_sizes()`, `log_sizes()` | learning-curve loop, metrics, CV, plots |

**Important: do NOT scale, one-hot encode or impute in data prep.** The pipeline does
that inside the model, so it is fitted on each training sample only (no leakage).

## 1. Loaders: `lab/data_<name>.py`
```python
def load_wine() -> tuple[pd.DataFrame, pd.Series]: ...
def load_power(max_rows=None) -> tuple[pd.DataFrame, pd.Series]: ...
```
- `X`: one row per data point, one column per feature.
  - numeric features have a **numeric dtype** (int / float),
  - categorical features have dtype `object`, `str` or `category`.
  The pipeline picks columns **by dtype**: numeric -> imputed, (squared,) scaled;
  everything else -> one-hot. So an integer that is really a label (e.g. hour of day)
  should be converted with `.astype("category")`.
- `y`: target Series, **no missing values** (drop those rows).
- Missing values in `X` are fine as `NaN` (the pipeline imputes them).
- No leakage columns and no ID columns.
- Read paths via `from lab import DATA_DIR` (respects the `LAB_DATA_DIR` env var).

## 2. Split: `lab/split.py`
```python
def train_val_split(X, y, task, seed=42):
    """task = "regression" | "classification". Returns X_train, X_val, y_train, y_val."""
```
- 20% validation, fixed seed, stratified on `y` for classification.
- Keep pandas types (DataFrame / Series) in the output.

## 3. Sampler: `lab/sampler.py`
```python
def sample_training_sets(y_train, sizes, n_repeats, task, seed=0):
    """Yield (size, repeat, idx) for every size and repeat."""

def linear_sizes(start, stop, step) -> list[int]   # 10, 20, ..., stop
def log_sizes(lo, hi, num) -> list[int]            # ~log-spaced unique ints lo..hi
```
- `idx` = **positional** indices into `X_train` / `y_train` (we use `.iloc[idx]`).
- Sampling **without replacement**, each `(size, repeat)` with its own reproducible
  seed (e.g. `np.random.SeedSequence([seed, size, repeat])`).
- Classification: stratified, and **every class must appear in every sample**,
  otherwise log-loss cannot be computed. Raise an error for sizes < number of classes.
- If `size >= len(y_train)`: return all rows (every repeat is then identical).
- A generator is fine; the loop consumes it once.

## 4. Dataset notes
- **wine:** class = column 1 (values 1-3), 13 numeric features, no missing values.
  Classification, stratified. Sizes e.g. `linear_sizes(10, len(X_train), 10)`.
- **household power** (`;`-separated, `?` = missing):
  - target `Global_active_power`; drop rows where it is missing.
  - drop `Global_intensity` (it is ~ power / voltage, i.e. leaks the target).
  - `Global_reactive_power`, `Voltage`, `Sub_metering_1..3` are numeric features.
  - `Date` / `Time` -> e.g. `hour`, `weekday`, `month`, made categorical
    (or keep `month` numeric, your call); drop the raw strings.
  - 2M rows: a `max_rows` option for quick local runs is handy. Sizes e.g.
    `log_sizes(10, len(X_train), 15)`.

## 5. How a notebook uses it
```python
from lab.data_wine import load_wine
from lab.split import train_val_split
from lab.sampler import sample_training_sets, linear_sizes
from lab.experiment import run_learning_curve, baseline_score, cross_validate_models
from lab.plotting import plot_learning_curve

X, y = load_wine()
X_train, X_val, y_train, y_val = train_val_split(X, y, "classification")
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
