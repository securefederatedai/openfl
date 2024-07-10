# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""openfl.utilities.data_splitters.data_splitter module."""
from abc import ABC, abstractmethod
from typing import Iterable, List, TypeVar

T = TypeVar("T")


class DataSplitter(ABC):
    """Base class for data splitting."""

    @abstractmethod
    def split(self, data: Iterable[T], num_collaborators: int) -> List[Iterable[T]]:
        """Split the data."""
        raise NotImplementedError
