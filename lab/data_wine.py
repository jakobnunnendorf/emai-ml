"""Wine dataset (UCI "Wine", file wine.data).

Small classification dataset: 178 wines from the same region in Italy, made
from three different cultivars (grape varieties). 13 numeric features from a
chemical analysis, target = cultivar (1, 2 or 3; 59 / 71 / 48 wines).
Raw file: data/wine/wine.data (no header, class in the first column, no
missing values).

With 3 classes, logistic regression is multinomial (softmax).

The loader only reads the data. It does NOT scale anything; that happens
inside the model pipeline (lab/models.py), fitted on the training sample only.
The features differ in scale by a factor of ~10,000 (Hue ~1, Proline ~1000),
so that scaling matters for the logistic regression solver.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from lab import DATA_DIR

# Column names from wine.names, section 7 ("Attribute Information").
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
    """Random 80/20 split, stratified on the cultivar.

    Every wine is a separate bottle, so the rows are independent and a random
    split is fine (no groups or time order to respect). With only ~36
    validation wines, a plain random split could easily leave one cultivar
    under-represented; stratifying keeps the class proportions (59 : 71 : 48)
    the same in training and validation, so every class is in both parts.
    Cross-validation can use the default StratifiedKFold for the same reason.

    Returns X_train, X_val, y_train, y_val (like train_test_split).
    """
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
