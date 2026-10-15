
## Datasets
- small reg: automobile (205 rows) -> tomi
- small class: wine (178 rows, 3 classes -> multinomial) -> Jakob
- big reg: household power (2M rows) -> Jakob
    - target? (Global_active_power?)
    - leakage: Global_intensity ~ power/voltage -> drop?
    - missing values = `?`
- big class: taxi (1.7M rows) -> Tomi
    - decided: label = CALL_TYPE (A/B/C), simple features (hour, weekday, duration,
      start/end coords, distance); ORIGIN_CALL / ORIGIN_STAND dropped (leak the label)
- subsample big ones?

## Prepare Data -> Jakob
- 20% -> Validation set (do not touch for training)
    - fixed seed, stratified for class?
- sampler: absolute sizes (10, 20, 30, ...), not % of train data
    - grid? linear for small, log for big (10 -> 10k)?
- repeat sampling with different seeds (3-times? more is cheap) -> mean ± std
- loaders for wine + household power (automobile + taxi: Tomi)
- exact interface: see `HANDOFF_DATA_PREP.md`

## Pipeline -> Tomi
- preprocessing: one-hot, missing values, scaling -> inside an sklearn Pipeline,
  so it is fitted on each training sample only (`lab/models.py`)
- models: linear, quadratic, logistic, quadratic_logistic, no regularization
    - quadratic = squares/products of NUMERIC columns only, one-hot columns not squared
    - p > n allowed (min-norm least squares), documented in `lab/models.py`
- learning-curve loop + MSE / log-loss + trivial baseline (`lab/experiment.py`)
- cross-validation, k=5 on the 80% train part (`lab/experiment.py`)
- shared plot style: median + 25-75% band, saved to `figures/` (`lab/plotting.py`)
- step 4 = full repeat of steps 1-3 for the classification datasets
- notebooks: `notebooks/automobile.ipynb`, `notebooks/taxi.ipynb`

## Implement models
### Linear Regression
- apply LR --> MSE
- plot MSE on training and validation vs. # samples
- explain behaviour

### Quadratic Regression
- repeat with quadratic regression
- automobile + one-hot -> ~2k features > n -> allow, or small ridge?

### Logistic Regression
- same experiment for class datasets, log-loss instead of MSE
- sklearn default L2 (C=1) -> turn off?
- small samples may miss a class -> stratify / `labels=` in log_loss
- step 4 "repeat the experiment" = only step 1, or also quadratic + CV?

## Cross-Validation
- only for: linear vs. quadratic on each reg dataset
- compare CV score with results from 1. and 2.
- k? (5 / 10), only on the 80% train part
- on full train part or per sample size?

## Metrics
- fixed: MSE (reg), log-loss (class)
- extras? (R², accuracy)

## Organization
- shared: loading, split, sampler, learning-curve loop, plotting
- one script per dataset
- same plot style / naming
- report format? who writes intro / shared parts? due 26 Oct
