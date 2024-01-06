# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime module FederatedRuntime class."""

from __future__ import annotations
from openfl.experimental.runtime import Runtime
from typing import TYPE_CHECKING, Type, List
if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator, Collaborator


class FederatedRuntime(Runtime):
    """Class for a federated runtime, derived from the Runtime class.

    Attributes:
        aggregator (Type[Aggregator]): The aggregator participant.
        collaborators (List[Type[Collaborator]]): The list of collaborator participants.
    """
        
    def __init__(
            self,
            aggregator: Type[Aggregator],
            collaborators: List[Type[Collaborator]] = None
    ) -> None:
        """Initializes the FederatedRuntime object with an aggregator and an optional list of collaborators.
        Use remote federated infrastructure to run the flow.
        
        Args:
            aggregator (Type[Aggregator]): The aggregator participant.
            collaborators (List[Type[Collaborator]], optional): The list of collaborator participants. Defaults to None.

        Raises:
            NotImplementedError: FederatedRuntime will be implemented in the future.
        """
        raise NotImplementedError("FederatedRuntime will be implemented in the future")
