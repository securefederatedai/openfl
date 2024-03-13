# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime package LocalRuntime class."""

from __future__ import annotations
from openfl.experimental.runtime import Runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator
    from openfl.experimental.interface import Collaborator

from typing import List
from typing import Type


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
