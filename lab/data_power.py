"""Household power dataset (UCI "Individual household electric power consumption").

Big regression dataset: one household near Paris, one row per MINUTE from
December 2006 to November 2010 (~2.07M rows).
Target: Global_active_power, the household's average power in that minute (kW).
Raw file: data/household_power_consumption.txt (";"-separated, "?" = missing).

Features we use (one row per minute):
    Global_reactive_power  kW, numeric
    Voltage                V, numeric
    Sub_metering_1..3      energy (Wh) of the kitchen / laundry room /
                           water heater + air conditioner in that minute
    hour, weekday, month   from Date / Time, categorical (one-hot): hour 23
                           is not "23x" hour 1, and the daily / yearly
                           patterns are not straight lines

Leakage: Global_intensity (A) is dropped. Power = voltage x current, so
Global_active_power ~ Voltage * Global_intensity / 1000 and the "prediction"
would just be a known formula. The sub-meterings are kept as agreed: they
measure only parts of the house, the rest (fridge, lights, electronics, ...)
is not metered, so they are strong but honest predictors of the total.

Missing values: ~26k minutes (1.25%) have "?" in every measurement (the meter
was off). Without a target we cannot use them, so they are dropped.

Split: BLOCKS OF TIME, see split_power. Cross-validation uses the same blocks
(whole weeks) as groups, see week_groups.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from lab import DATA_DIR

TARGET = "Global_active_power"
NUMERIC = ["Global_reactive_power", "Voltage", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
CATEGORICAL = ["hour", "weekday", "month"]
LEAKY = ["Global_intensity"]


def load_power(path=None, max_rows=None, seed=0):
    """Return (X, y, time) for the household power dataset.

    time is the timestamp of each minute. It is NOT a feature; it is returned
    separately so split_power / week_groups can keep whole weeks together.

    max_rows: use a random subset of this many minutes (handy for quick runs);
              None = all ~2.05M minutes. Reading the full file takes ~10 s.
    """
    path = path or DATA_DIR / "household_power_consumption.txt"
    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)

    time = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    X = df[NUMERIC].astype(float)
    X["hour"] = time.dt.hour.astype("category")
    X["weekday"] = time.dt.weekday.astype("category")  # 0 = Monday
    X["month"] = time.dt.month.astype("category")

    y = df[TARGET].astype(float)
    assert not set(LEAKY) & set(X.columns), "leaky columns must not be features"
    return X, y, time


def week_groups(time):
    """Calendar week of each minute ("2007-W03", ...): the independent unit for
    splitting and cross-validation (sklearn GroupShuffleSplit / GroupKFold)."""
    iso = time.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def split_power(X, y, time, val_fraction=0.2, seed=42):
    """Split by WHOLE WEEKS: a random ~20% of the weeks become the validation set.

    Neighbouring minutes are almost identical (the fridge is still running,
    the oven still on). A random split at the minute level puts the minute
    before and after every validation minute into the training data, so the
    validation set is not really "new" data. Keeping whole weeks on one side
    removes that: the model has to predict days it has never seen.
    (For our linear models the effect turns out to be small, because ~50
    coefficients fitted on 1.6M rows cannot memorize single minutes; a
    flexible model such as k-nearest-neighbours would profit a lot. See
    split_power_random and the comparison in notebooks/power.ipynb.)

    Why random weeks and not simply the last 20% in time? The data cover four
    years; the last 20% would be only summer and autumn 2010, so validation
    would mix "unseen days" with "only one season". Random weeks cover every
    season in both parts. Weeks rather than days because consecutive days of
    the same week (holidays, someone away) are still strongly related.

    Returns X_train, X_val, y_train, y_val (like train_test_split).
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=week_groups(time)))
    return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]


def split_power_random(X, y, val_fraction=0.2, seed=42):
    """NAIVE random split at the minute level. Do NOT use it for the results;
    it only exists for the comparison with split_power in the report (does the
    leakage between neighbouring minutes flatter the validation error?)."""
    return train_test_split(X, y, test_size=val_fraction, random_state=seed)
