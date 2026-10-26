# IML Lab 1: learning curves

Learning curves for linear and quadratic regression (automobile, household power) and for
logistic regression (wine, taxi), plus cross-validation. Assignment: `Lab1.pdf`.

```
lab/         shared code: data loaders + splits, sampler, models, experiments, plots
notebooks/   one notebook per dataset, with the results and their explanation
results/     CSV of every run
figures/     plots
```

| Notebook | Dataset (UCI) | Task |
|---|---|---|
| `notebooks/automobile.ipynb` | Automobile | small regression |
| `notebooks/power.ipynb` | Individual household electric power consumption | large regression |
| `notebooks/wine.ipynb` | Wine | small classification |
| `notebooks/taxi.ipynb` | Taxi Service Trajectory (ECML PKDD 2015) | large classification |

## Running
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/automobile.ipynb
```
The raw data go into `data/` (or set `LAB_DATA_DIR`):
```
data/automobile/imports-85.data
data/wine/wine.data
data/household_power_consumption.txt
data/taxi/train.csv
```
For the taxi data, unzip `train.csv` from the downloaded archive (it is a zip inside a zip).
The taxi notebook uses a random subset of 20,000 trips; set `RUN_FULL = True` for all trips.
