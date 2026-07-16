"""
Dataset loaders — all formats in one file.
"""

import zipfile
import tempfile
from pathlib import Path

import pandas as pd

from .base_loader import BaseLoader


class CSVLoader(BaseLoader):
    def load(self, source): return pd.read_csv(source)


class JSONLoader(BaseLoader):
    def load(self, source): return pd.read_json(source)


class ExcelLoader(BaseLoader):
    def load(self, source): return pd.read_excel(source)


class ZIPLoader(BaseLoader):
    def load(self, source):
        temp = tempfile.mkdtemp()
        with zipfile.ZipFile(source) as zf:
            zf.extractall(temp)
        csv_files = list(Path(temp).rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV found inside ZIP")
        return pd.read_csv(csv_files[0])


class URLLoader(BaseLoader):
    """Loads a dataset from a URL. Detects format by file extension."""
    def load(self, source: str):
        lower = source.lower().split("?")[0]  # strip query params for ext check
        if lower.endswith(".json"):
            return pd.read_json(source)
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(source)
        # Default: treat as CSV (covers .csv and bare URLs)
        return pd.read_csv(source)
