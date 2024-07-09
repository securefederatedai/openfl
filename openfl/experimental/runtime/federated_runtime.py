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

""" openfl.experimental.runtime package LocalRuntime class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openfl.experimental.runtime.runtime import Runtime

if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator
    from openfl.experimental.interface import Collaborator

from typing import List, Type


class FederatedRuntime(Runtime):

    def __init__(
        self,
        aggregator: str = None,
        collaborators: List[str] = None,
        **kwargs,
    ) -> None:
        """
        Use single node to run the flow

        Args:
            aggregator:    Name of the aggregator.
            collaborators: List of collaborator names.

        Returns:
            None
        """
        super().__init__()
        if aggregator is not None:
            self.aggregator = aggregator

        if collaborators is not None:
            self.collaborators = collaborators

    @property
    def aggregator(self) -> str:
        """Returns name of _aggregator"""
        return self._aggregator

    @aggregator.setter
    def aggregator(self, aggregator_name: Type[Aggregator]):
        """Set LocalRuntime _aggregator"""
        self._aggregator = aggregator_name

    @property
    def collaborators(self) -> List[str]:
        """
        Return names of collaborators. Don't give direct access to private attributes
        """
        return self.__collaborators

    @collaborators.setter
    def collaborators(self, collaborators: List[Type[Collaborator]]):
        """Set LocalRuntime collaborators"""
        self.__collaborators = collaborators

    def __repr__(self):
        return "FederatedRuntime"
