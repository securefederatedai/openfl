# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


""" openfl.experimental.runtime package LocalRuntime class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openfl.experimental.runtime.runtime import Runtime

if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator
    from openfl.experimental.interface import Collaborator

from typing import List, Type


class FederatedRuntime(Runtime):
    """Class for a federated runtime, derived from the Runtime class.

    Attributes:
        aggregator (Type[Aggregator]): The aggregator participant.
        collaborators (List[Type[Collaborator]]): The list of collaborator
            participants.
    """

    def __init__(
        self,
        aggregator: str = None,
        collaborators: List[str] = None,
        **kwargs,
    ) -> None:
        """Initializes the FederatedRuntime object.

        Use single node to run the flow.

        Args:
            aggregator (str, optional): Name of the aggregator. Defaults to
                None.
            collaborators (List[str], optional): List of collaborator names.
                Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        if aggregator is not None:
            self.aggregator = aggregator

        if collaborators is not None:
            self.collaborators = collaborators

    @property
    def aggregator(self) -> str:
        """Returns name of _aggregator."""
        return self._aggregator

    @aggregator.setter
    def aggregator(self, aggregator_name: Type[Aggregator]):
        """Set LocalRuntime _aggregator.

        Args:
            aggregator_name (Type[Aggregator]): The name of the aggregator to
                set.
        """
        self._aggregator = aggregator_name

    @property
    def collaborators(self) -> List[str]:
        """Return names of collaborators.

        Don't give direct access to private attributes.

        Returns:
            List[str]: The names of the collaborators.
        """
        return self.__collaborators

    @collaborators.setter
    def collaborators(self, collaborators: List[Type[Collaborator]]):
        """Set LocalRuntime collaborators.

        Args:
            collaborators (List[Type[Collaborator]]): The list of
                collaborators to set.
        """
        self.__collaborators = collaborators

    def __repr__(self):
        return "FederatedRuntime"
