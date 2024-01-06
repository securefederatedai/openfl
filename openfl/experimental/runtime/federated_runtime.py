# Copyright (C) 2020-2023 Intel Corporation
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
        collaborators (List[Type[Collaborator]]): The list of collaborator participants.
    """
        
    def __init__(
        self,
        aggregator: str = None,
        collaborators: List[str] = None,
        **kwargs,
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
