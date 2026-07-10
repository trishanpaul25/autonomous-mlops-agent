"""
Dataset loaders.
"""

from .CSVLoader import CSVLoader
from .ExceLoader import ExcelLoader
from .JSONLoader import JSONLoader
from .URLLoader import URLLoader
from .ZIPLoader import ZIPLoader

# Backward-compatible aliases for older imports.
csv_loader = CSVLoader
excel_loader = ExcelLoader
json_loader = JSONLoader
url_loader = URLLoader
zip_loader = ZIPLoader

__all__ = [
    "CSVLoader",
    "ExcelLoader",
    "JSONLoader",
    "URLLoader",
    "ZIPLoader",
    "csv_loader",
    "excel_loader",
    "json_loader",
    "url_loader",
    "zip_loader",
]