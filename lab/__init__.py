"""Shared code for IML Lab 1 (learning curves, linear / quadratic / logistic regression)."""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

# Where the raw datasets live. Defaults to <repo>/data; override with the
# LAB_DATA_DIR environment variable (e.g. on the JSI servers).
DATA_DIR = Path(os.environ.get("LAB_DATA_DIR", REPO_DIR / "data"))

# Outputs of the experiments (committed, used in the report).
RESULTS_DIR = REPO_DIR / "results"
FIGURES_DIR = REPO_DIR / "figures"
