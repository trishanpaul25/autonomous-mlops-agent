import pandas as pd

from .base_loader import BaseLoader


class CSVLoader(BaseLoader):
    def load(self, source):
        return pd.read_csv(source)