import pandas as pd

from .base_loader import BaseLoader


class ExcelLoader(BaseLoader):
    def load(self, source):
        return pd.read_excel(source)