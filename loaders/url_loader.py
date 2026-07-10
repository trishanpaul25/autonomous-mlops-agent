import pandas as pd

from loaders.base_loader import BaseLoader


class URLLoader(BaseLoader):
    def load(self, source):
        return pd.read_csv(source)