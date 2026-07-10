from abc import ABC, abstractmethod

class BaseLoader(ABC):
    @abstractmethod
    def load(self, source):
        pass