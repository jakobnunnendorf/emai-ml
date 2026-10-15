"""Automobile dataset (UCI "Automobile", file imports-85.data).

Small regression dataset: 205 cars, 25 features, target = price.
Raw file: data/automobile/imports-85.data (no header, "?" = missing value).

The loader only cleans the data. It does NOT impute, scale or one-hot encode;
that happens inside the model pipeline (lab/models.py), fitted on the
training sample only.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

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


def split_automobile(X, y, seed=42):
    """Random 80/20 split, stratified on price.

    Prices are skewed (median ~$10k, a handful of cars at $30-45k) and MSE
    squares the errors, so with only ~40 validation cars it matters a lot how
    many expensive cars end up there. We cut the prices into 5 equally full
    bins (quintiles) and stratify on them: training and validation then cover
    the same price range.

    Note: the dataset lists several trims of the same model (near-identical
    rows). A random split can put trims of one model on both sides, which
    makes the validation error slightly optimistic. Grouping by model would
    avoid that but answers a different question ("how well on a car model we
    have never seen"), so we keep the random split and mention it in the report.

    Returns X_train, X_val, y_train, y_val (like train_test_split).
    """
    price_bins = pd.qcut(y, q=5, labels=False)
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=price_bins)
