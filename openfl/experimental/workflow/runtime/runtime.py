# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.workflow.runtime module Runtime class."""

from typing import Callable, List

from openfl.experimental.workflow.interface.fl_spec import FLSpec
from openfl.experimental.workflow.interface.participants import Aggregator, Collaborator


class AttributeValidationMeta(type):
    """
    Metaclass that enforces validation rules on class attributes.

    This metaclass ensures that `prohibited_data_types` and `allowed_data_types`
    are lists of strings and that both cannot be set at the same time.

    Example:
        class MyClass(metaclass=AttributeValidationMeta):
            pass

        MyClass.prohibited_data_types = ["int", "float"]  # Valid
        MyClass.allowed_data_types = ["str", "bool"]  # Valid
        MyClass.prohibited_data_types = "int"  # Raises TypeError
        MyClass.allowed_data_types = 42  # Raises TypeError
    """

    def __setattr__(cls, name, value):
        """
        Validates and sets class attributes.

        Ensures that `prohibited_data_types` and `allowed_data_types`, when assigned,
        are lists of strings and that they are not used together.

        Args:
            name (str): The attribute name being set.
            value (any): The value to be assigned to the attribute.

        Raises:
            TypeError: If `prohibited_data_types` or `allowed_data_types` is not a list
                or contains non-string elements.
            ValueError: If both `prohibited_data_types` and `allowed_data_types` are set.
        """
        if name in {"prohibited_data_types", "allowed_data_types"}:
            if not isinstance(value, list):
                raise TypeError(f"'{name}' must be a list, got {type(value).__name__}")
            if not all(isinstance(item, str) for item in value):
                raise TypeError(f"All elements of '{name}' must be strings")

            # Ensure both attributes are not set at the same time
            other_name = (
                "allowed_data_types" if name == "prohibited_data_types" else "prohibited_data_types"
            )
            if getattr(cls, other_name, []):  # Check if the other attribute is already set
                raise ValueError(
                    "Cannot set both 'prohibited_data_types' and 'allowed_data_types'."
                )

        super().__setattr__(name, value)


class Runtime(metaclass=AttributeValidationMeta):
    """
    Base class for federated learning runtimes.
    This class serves as an interface for runtimes that execute FLSpec flows.

    Attributes:
        prohibited_data_types (list): A list of data types that are prohibited from being
            transmitted over the network.
        allowed_data_types (list): A list of data types that are explicitly allowed to be
            transmitted over the network.

    Notes:
        - Either `prohibited_data_types` or `allowed_data_types` may be specified.
        - If neither is specified, all data types are allowed to be transmitted.
        - If both are specified, a `ValueError` will be raised.
    """

    prohibited_data_types = []
    allowed_data_types = []

    def __init__(self):
        """Initializes the Runtime object.

        This serves as a base interface for runtimes that can run FLSpec flows.
        """
        pass

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
