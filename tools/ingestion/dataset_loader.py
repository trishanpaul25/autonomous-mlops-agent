"""
tools/ingestion/dataset_loader.py

Dataset Loader Factory.

Selects the appropriate loader based on the dataset source type
and returns a pandas DataFrame.
"""

from tools.loaders import (
    CSVLoader,
    ExcelLoader,
    JSONLoader,
    URLLoader,
    ZIPLoader,
)

from pathlib import Path


class DatasetLoader:
    """
    Factory class responsible for selecting the correct loader.
    """

    def __init__(self):

        self.loaders = {
            "csv": CSVLoader(),
            "excel": ExcelLoader(),
            "json": JSONLoader(),
            "url": URLLoader(),
            "zip": ZIPLoader(),
        }

    def load(
        self,
        source_type: str,
        source: str,
    ):
        """
        Load the dataset and return a pandas DataFrame.

        Parameters
        ----------
        source_type : str
            Dataset source type.

        source : str
            File path, URL, or dataset identifier.

        Returns
        -------
        pandas.DataFrame
        """

        if source_type not in self.loaders:

            raise ValueError(
                f"Unsupported dataset source type: {source_type}"
            )

        loader = self.loaders[source_type]

        if source_type in {"csv", "excel", "json", "zip"}:
            source_path = Path(source)

            if not source_path.is_absolute():
                project_root = Path(__file__).resolve().parents[2]
                candidate = project_root / source_path

                if candidate.exists():
                    source = str(candidate)

        return loader.load(source)