import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

# raw data; override with LAB_DATA_DIR
DATA_DIR = Path(os.environ.get("LAB_DATA_DIR", REPO_DIR / "data"))

RESULTS_DIR = REPO_DIR / "results"
FIGURES_DIR = REPO_DIR / "figures"
