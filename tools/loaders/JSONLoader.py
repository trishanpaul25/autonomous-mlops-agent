import pandas as pd

from .base_loader import BaseLoader


class JSONLoader(BaseLoader):
    def load(self, source):
        return pd.read_json(source)