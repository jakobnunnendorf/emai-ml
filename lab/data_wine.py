"""UCI Wine: 178 wines, 13 numeric features, target = cultivar (1, 2, 3)."""

import pandas as pd
from sklearn.model_selection import train_test_split

from lab import DATA_DIR

# from wine.names
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline",
]
TARGET = "cultivar"


def load_wine(path=None):
    """Return (X, y): X has the 13 numeric features, y the cultivar (1, 2, 3)."""
    path = path or DATA_DIR / "wine" / "wine.data"
    df = pd.read_csv(path, header=None, names=[TARGET] + FEATURES)

    X = df[FEATURES].astype(float)
    y = df[TARGET]
    return X, y


def split_wine(X, y, seed=42):
    """Random 80/20 split, stratified on the cultivar (rows are independent,
    stratifying keeps all classes in both parts)."""
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
