"""
retriever: this abstracts out the various different data retrieval methods available
"""

from abc import ABC, abstractmethod

class AbstractRetriever(ABC):
    """Docstring for AbstractRetriever. """

    @abstractmethod
    def retrieve(self):
        """returns an iterator of 3d numpy arrays that act as video"""
        pass

    def reset(self):
        """resets the retriever"""
        pass
