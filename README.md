# IML Lab 1: learning curves for linear, quadratic and logistic regression

Assignment: `Lab1.pdf` (due 26 Oct). Team plan: `initial-plan.md`.
Interface between data prep and the pipeline: `HANDOFF_DATA_PREP.md`.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
No GPU needed; everything runs on the CPU.

## Data (not in git)
Put the raw files in `data/` (or point the `LAB_DATA_DIR` environment variable elsewhere):
```
data/automobile/imports-85.data
data/wine/wine.data
data/household_power_consumption.txt
data/taxi/train.csv            # see below
```
The taxi zip contains another zip. Unpack `train.csv` once:
```bash
mkdir -p data/taxi
unzip -j "data/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip" train.csv.zip -d data/taxi
unzip data/taxi/train.csv.zip -d data/taxi && rm data/taxi/train.csv.zip
```
On first use `load_taxi()` builds `data/taxi_features.parquet` (about a minute).

## Layout
```
lab/                 shared Python code
  models.py          the four model kinds as sklearn Pipelines (preprocessing inside)
  experiment.py      learning-curve loop, metrics, baseline, cross-validation
  plotting.py        shared plot style
  data_*.py          one loader per dataset
  temp_prep.py       temporary split + sampler until lab/split.py, lab/sampler.py exist
notebooks/           one notebook per dataset (experiments + explanations)
results/             CSV of every experiment run
figures/             PNG plots for the report
```

## Running
Open a notebook in Jupyter (`jupyter lab`) or run it headless:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/automobile.ipynb
```
`notebooks/taxi.ipynb` uses a 20k-trip subset by default. For the full run set
`RUN_FULL = True` in its config cell and run it on the JSI servers with the same command.
