# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.workflow.runtime module Runtime class."""

from typing import Callable, List, Optional

from openfl.experimental.workflow.interface.fl_spec import FLSpec
from openfl.experimental.workflow.interface.participants import Aggregator, Collaborator


class Runtime:
    def __init__(self, prohibited_data_types: Optional[List[str]] = None):
        """Initializes the Runtime object.

        This serves as a base interface for runtimes that can run FLSpec flows.

        Args:
            prohibited_data_types (Optional[List[str]]): A list of data types that are not allowed to be sent
                through the network. Defaults to an empty list if not provided.
        """
        self.prohibited_data_types = prohibited_data_types or []

    @property
    def prohibited_data_types(self) -> List[str]:
        """Return the prohibited data types for the runtime."""
        return self._prohibited_data_types

    @prohibited_data_types.setter
    def prohibited_data_types(self, value: List[str]):
        """Set the prohibited data types for the runtime."""
        if not isinstance(value, list):
            raise TypeError("prohibited_data_types must be a list of strings.")
        if not all(isinstance(item, str) for item in value):
            raise ValueError("All items in prohibited_data_types must be strings.")
        self._prohibited_data_types = value

    @property
    def aggregator(self):
        """Returns the name of the aggregator.

        Raises:
            NotImplementedError: If the method is not implemented in a
                subclass.
        """
        raise NotImplementedError

    @aggregator.setter
    def aggregator(self, aggregator: Aggregator):
        """Sets the aggregator of the Runtime.

        Args:
            aggregator (Aggregator): The aggregator to be set.

        Raises:
            NotImplementedError: If the method is not implemented in a
                subclass.
        """
        raise NotImplementedError

    @property
    def collaborators(self):
        """Return the names of the collaborators. Don't give direct access to
        private attributes.

        Raises:
            NotImplementedError: If the method is not implemented in a
                subclass.
        """
        raise NotImplementedError

    @collaborators.setter
    def collaborators(self, collaborators: List[Collaborator]):
        """Sets the collaborators of the Runtime.

        Args:
            collaborators (List[Collaborator]): The collaborators to be set.

        Raises:
            NotImplementedError: If the method is not implemented in a
                subclass.
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
            flspec_obj (FLSpec): Reference to the FLSpec (flow) object.
                Contains information about task sequence, flow attributes,
                that are needed to execute a future task.
            f (Callable): The next task to be executed within the flow.
            parent_func (Callable): The prior task executed in the flow.
            instance_snapshot (List[FLSpec], optional): A prior FLSpec state
                that needs to be restored from (i.e. restoring aggregator
                state after collaborator execution).
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If the method is not implemented in a
                subclass.
        """
        raise NotImplementedError
