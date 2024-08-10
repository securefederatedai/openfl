# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.utilities.data_splitters.data_splitter module."""
from abc import ABC, abstractmethod
from typing import Iterable, List, TypeVar

T = TypeVar("T")


class DataSplitter(ABC):
    """Base class for data splitting.

    This class should be subclassed when creating specific data splitter
    classes.
    """

    @abstractmethod
    def split(self, data: Iterable[T], num_collaborators: int) -> List[Iterable[T]]:
        """Split the data into a specified number of parts.

        Args:
            data (Iterable[T]): The data to be split.
            num_collaborators (int): The number of parts to split the data
                into.

        Returns:
            List[Iterable[T]]: The split data.

        Raises:
            NotImplementedError: This is an abstract method and must be
                overridden in a subclass.
        """
        raise NotImplementedError
