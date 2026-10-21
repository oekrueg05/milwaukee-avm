"""Project paths. Data lives outside the repo (and outside any synced folder).

Override the data location with the AVM_DATA_DIR environment variable.
"""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AVM_DATA_DIR", Path.home() / "data" / "milwaukee-avm"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = RAW_DIR / "manifest.csv"

REPORTS_DIR = REPO_DIR / "reports"
# Parcel-level outputs with addresses or tax keys (divergence lists, the private
# review page): written outside the repo, never committed.
PRIVATE_REPORTS_DIR = DATA_DIR / "reports_private"

CKAN_API = "https://data.milwaukee.gov/api/3/action"
