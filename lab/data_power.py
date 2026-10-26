"""UCI Individual household electric power consumption: one row per minute,
Dec 2006 - Nov 2010 (~2M rows). Target = Global_active_power (kW).

Global_intensity is dropped: active power ~ voltage * intensity, so it leaks
the target. Minutes with no measurement (~1.25%) are dropped.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from lab import DATA_DIR

TARGET = "Global_active_power"
NUMERIC = ["Global_reactive_power", "Voltage", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
CATEGORICAL = ["hour", "weekday", "month"]
LEAKY = ["Global_intensity"]


def load_power(path=None, max_rows=None, seed=0):
    """Return (X, y, time). time is only used for splitting, not a feature.
    max_rows: random subset for quick runs."""
    path = path or DATA_DIR / "household_power_consumption.txt"
    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)

    time = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    X = df[NUMERIC].astype(float)
    X["hour"] = time.dt.hour.astype("category")
    X["weekday"] = time.dt.weekday.astype("category")
    X["month"] = time.dt.month.astype("category")

    y = df[TARGET].astype(float)
    assert not set(LEAKY) & set(X.columns), "leaky columns must not be features"
    return X, y, time


def week_groups(time):
    """Calendar week of each minute, e.g. "2007-W03" (groups for split and CV)."""
    iso = time.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def split_power(X, y, time, val_fraction=0.2, seed=42):
    """Random 20% of whole calendar weeks as validation.

    Neighbouring minutes are nearly identical, so a minute-level random split
    would put near-copies of validation rows into training. Random weeks
    instead of the last 20% in time so that all seasons are in both parts.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=week_groups(time)))
    return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]


def split_power_random(X, y, val_fraction=0.2, seed=42):
    """Minute-level random split, only for the comparison in notebooks/power.ipynb."""
    return train_test_split(X, y, test_size=val_fraction, random_state=seed)
