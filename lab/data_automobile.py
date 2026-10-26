"""UCI Automobile (imports-85.data): 205 cars, 25 features, target = price."""

import pandas as pd
from sklearn.model_selection import train_test_split

from lab import DATA_DIR

# from imports-85.names
COLUMNS = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
    "price",
]
TARGET = "price"

# counts stored as words
WORD_TO_NUMBER = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}


def load_automobile(path=None):
    """Return (X, y). Drops the 4 cars without a price; other missing values stay NaN."""
    path = path or DATA_DIR / "automobile" / "imports-85.data"
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    for column in ["num-of-doors", "num-of-cylinders"]:
        df[column] = df[column].map(WORD_TO_NUMBER)

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return X, y


def split_automobile(X, y, seed=42):
    """Random 80/20 split, stratified on price quintiles.

    Prices are skewed and the validation set is small (~40 cars), so without
    stratifying the MSE depends a lot on how many expensive cars land there.
    """
    price_bins = pd.qcut(y, q=5, labels=False)
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=price_bins)
