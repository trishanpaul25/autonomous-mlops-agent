"""
Base class for all dataset loaders.
"""

from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """

    @abstractmethod
    def load(self, source):
        """
        Load dataset and return a pandas DataFrame.
        """
        pass