import zipfile
import tempfile
from pathlib import Path

import pandas as pd

from loaders.base_loader import BaseLoader


class ZIPLoader(BaseLoader):
    def load(self, source):
        temp = tempfile.mkdtemp()

        with zipfile.ZipFile(source) as zip_ref:
            zip_ref.extractall(temp)

        csv_files = list(Path(temp).rglob("*.csv"))

        if len(csv_files) == 0:
            raise Exception("No CSV found inside ZIP")

        return pd.read_csv(csv_files[0])