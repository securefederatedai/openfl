"""openfl.utilities.data_splitters.data_splitter module."""
from abc import ABC, abstractmethod
from typing import Iterable, List, TypeVar

T = TypeVar('T')


class DataSplitter(ABC):
    """Base class for data splitting."""

    @abstractmethod
    def split(self, data: Iterable[T], num_collaborators: int) -> List[Iterable[T]]:
        """Split the data."""
        raise NotImplementedError
