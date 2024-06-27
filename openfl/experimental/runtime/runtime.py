# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
""" openfl.experimental.runtime module Runtime class."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator, Collaborator, FLSpec

from typing import Callable, List


class Runtime:

    def __init__(self):
        """Initializes the Runtime object. This serves as a base interface for runtimes that can run FLSpec flows."""
        pass

    @property
    def aggregator(self):
        """Returns the name of the aggregator.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @aggregator.setter
    def aggregator(self, aggregator: Aggregator):
        """Sets the aggregator of the Runtime.

        Args:
            aggregator (Aggregator): The aggregator to be set.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @property
    def collaborators(self):
        """Return the names of the collaborators. Don't give direct access to private attributes

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @collaborators.setter
    def collaborators(self, collaborators: List[Collaborator]):
        """Sets the collaborators of the Runtime.

        Args:
            collaborators (List[Collaborator]): The collaborators to be set.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def execute_task(
        self,
        flspec_obj: FLSpec,
        f: Callable,
        parent_func: Callable,
        instance_snapshot: List[FLSpec] = [],
        **kwargs,
    ):
        """Performs the execution of a task as defined by the implementation 
        and underlying backend (single_process, ray, etc).

        Args:
            flspec_obj (FLSpec): Reference to the FLSpec (flow) object. Contains information
                about task sequence, flow attributes, that are needed to execute a future task.
            f (Callable): The next task to be executed within the flow.
            parent_func (Callable): The prior task executed in the flow.
            instance_snapshot (List[FLSpec], optional): A prior FLSpec state that needs to be 
            restored from (i.e. restoring aggregator state after collaborator execution).
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
