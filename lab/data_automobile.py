"""Automobile dataset (UCI "Automobile", file imports-85.data).

Small regression dataset: 205 cars, 25 features, target = price.
Raw file: data/automobile/imports-85.data (no header, "?" = missing value).

The loader only cleans the data. It does NOT impute, scale or one-hot encode;
that happens inside the model pipeline (lab/models.py), fitted on the
training sample only.
"""

import pandas as pd

from lab import DATA_DIR

# Column names from imports-85.names, section 7 ("Attribute Information").
COLUMNS = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
    "price",
]
TARGET = "price"

# Two columns store counts as words. Turning them into numbers lets the model
# treat them as quantities (and include them in the quadratic expansion).
WORD_TO_NUMBER = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}


def load_automobile(path=None):
    """Return (X, y): X is a DataFrame of features, y the price Series.

    - "?" is read as NaN (missing).
    - The 4 cars without a price are dropped (we cannot learn from them).
    - Missing feature values stay NaN; the pipeline imputes them.
    """
    path = path or DATA_DIR / "automobile" / "imports-85.data"
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    for column in ["num-of-doors", "num-of-cylinders"]:
        df[column] = df[column].map(WORD_TO_NUMBER)

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return X, y
