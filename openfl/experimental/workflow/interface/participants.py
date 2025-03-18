# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.workflow.interface.participants module."""

from typing import Any, Callable, Dict, Optional


class Participant:
    """Class for a participant.

    Attributes:
        private_attributes (dict): The private attributes of the participant.
        _name (str): The name of the participant.
    """

    def __init__(self, name: str = ""):
        """Initializes the Participant object with an optional name.

        Args:
            name (str, optional): The name of the participant. Defaults to "".
        """
        self.private_attributes = {}
        self._name = name

    @property
    def name(self):
        """Returns the name of the participant.

        Returns:
            str: The name of the participant.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of the participant.

        Args:
            name (str): The name to be set.
        """
        self._name = name

    def get_name(self) -> str:
        """Gets the name of Participant (aggregator or collaborator).

        Returns:
            str: The name of the aggregator or collaborator.
        """
        return self._name

    def initialize_private_attributes(self, private_attrs: Dict[Any, Any] = None) -> None:
        """Initialize private attributes of Participant (aggregator or collaborator)
        by invoking the callable specified by user."""
        if self.private_attributes_callable is not None:
            self.private_attributes = self.private_attributes_callable(**self.kwargs)
        elif private_attrs:
            self.private_attributes = private_attrs

    def _set_private_attrs_to_clone(self, clone: Any) -> None:
        """Set private attributes to FLSpec clone before
        transitioning from aggregator to collaborator or vice versa

        Args:
            clone (Any): The clone to set attributes to.
        """
        # set private attributes as clone attributes
        for name, attr in self.private_attributes.items():
            setattr(clone, name, attr)

    def _delete_private_attrs_from_clone(self, clone: Any) -> None:
        """Remove private attributes from FLSpec clone before
        transitioning from collaborator to aggregator or vice versa

        Args:
            clone (Any): The clone to remove attributes from.
        """
        # Update  private attributes by taking latest
        # parameters from clone, then delete attributes from clone.
        for attr_name in self.private_attributes:
            if hasattr(clone, attr_name):
                self.private_attributes.update({attr_name: getattr(clone, attr_name)})
                delattr(clone, attr_name)

    def private_attributes(self, attrs: Dict[str, Any]) -> None:
        """Set the private attributes of the participant. These attributes will
        only be available within the tasks performed by the participants and
        will be filtered out prior to the task's state being transferred.

        Args:
            attrs (Dict[str, Any]): dictionary of ATTRIBUTE_NAME (str) ->
                object that will be accessible within the participant's task.

                Example:
                {'train_loader' : torch.utils.data.DataLoader(...)}

                In any task performed by this participant performed within the
                flow, this attribute could be referenced with self.train_loader
        """
        self.private_attributes = attrs


class Collaborator(Participant):
    """Class for a collaborator participant, derived from the Participant
    class.

    Attributes:
        name (str): Name of the collaborator.
        private_attributes_callable (Callable): A function which returns
            collaborator private attributes for each collaborator.
        num_cpus (int): Specifies how many cores to use for the collaborator
            step execution.
        num_gpus (float): Specifies how many GPUs to use to accelerate the
            collaborator step execution.
        kwargs (dict): Parameters required to call private_attributes_callable
            function.
    """

    def __init__(
        self,
        name: str = "",
        private_attributes_callable: Callable = None,
        num_cpus: int = 0,
        num_gpus: int = 0.0,
        **kwargs,
    ):
        """Initializes the Collaborator object.

        Create collaborator object with custom resources and a callable
        function to assign private attributes.

        Args:
            name (str, optional): Name of the collaborator. Defaults to "".
            private_attributes_callable (Callable, optional): A function which
                returns collaborator private attributes for each collaborator.
                In case private_attributes are not required this can be
                omitted. Defaults to None.
            num_cpus (int, optional): Specifies how many cores to use for the
                collaborator step execution. This will only be used if backend
                is set to ray. Defaults to 0.
            num_gpus (float, optional): Specifies how many GPUs to use to
                accelerate the collaborator step execution. This will only be
                used if backend is set to ray. Defaults to 0.0.
            **kwargs (dict): Parameters required to call
                private_attributes_callable function. The key of the
                dictionary must match the arguments to the
                private_attributes_callable. Defaults to {}.
        """
        super().__init__(name=name)
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.kwargs = kwargs

        if private_attributes_callable is None:
            self.private_attributes_callable = private_attributes_callable
        else:
            if not callable(private_attributes_callable):
                raise Exception("private_attributes_callable  parameter must be a callable")
            else:
                self.private_attributes_callable = private_attributes_callable

    def execute_func(self, ctx: Any, f_name: str, callback: Callable) -> Any:
        """Execute remote function f.

        Args:
            ctx (Any): The context to execute the function in.
            f_name (str): The name of the function to execute.
            callback (Callable): The callback to execute after the function.

        Returns:
            Any: The result of the function execution.
        """
        self._set_private_attrs_to_clone(ctx)

        callback(ctx, f_name)

        self._delete_private_attrs_from_clone(ctx)

        return ctx


class Aggregator(Participant):
    """Class for an aggregator participant, derived from the Participant
    class."""

    def __init__(
        self,
        name: str = "",
        private_attributes_callable: Callable = None,
        num_cpus: int = 0,
        num_gpus: int = 0.0,
        **kwargs,
    ):
        """Initializes the Aggregator object.

        Create aggregator object with custom resources and a callable
        function to assign private attributes.

        Args:
            name (str, optional): Name of the aggregator. Defaults to "".
            private_attributes_callable (Callable, optional): A function which
                returns aggregator private attributes. In case
                private_attributes are not required this can be omitted.
                Defaults to None.
            num_cpus (int, optional): Specifies how many cores to use for the
                aggregator step execution. This will only be used if backend
                is set to ray. Defaults to 0.
            num_gpus (float, optional): Specifies how many GPUs to use to
                accelerate the aggregator step execution. This will only be
                used if backend is set to ray. Defaults to 0.0.
            **kwargs: Parameters required to call private_attributes_callable
                function. The key of the dictionary must match the arguments
                to the private_attributes_callable. Defaults to {}.
        """
        super().__init__(name=name)
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.kwargs = kwargs

        if private_attributes_callable is None:
            self.private_attributes_callable = private_attributes_callable
        else:
            if not callable(private_attributes_callable):
                raise Exception("private_attributes_callable parameter must be a callable")
            else:
                self.private_attributes_callable = private_attributes_callable

    def execute_func(
        self, ctx: Any, f_name: str, callback: Callable, clones: Optional[Any] = None
    ) -> Any:
        """Executes remote function f.

        Args:
            ctx (Any): The context to execute the function in.
            f_name (str): The name of the function to execute.
            callback (Callable): The callback to execute after the function.
            clones (Optional[Any], optional): The clones to use in the
                function. Defaults to None.

        Returns:
            Any: The result of the function execution.
        """
        self._set_private_attrs_to_clone(ctx)

        if clones is not None:
            callback(ctx, f_name, clones)
        else:
            callback(ctx, f_name)

        self._delete_private_attrs_from_clone(ctx)

        return ctx
