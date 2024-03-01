# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.interface.participants module."""

from typing import Dict, Any
from typing import Callable, Optional


class Participant:
    def __init__(self, name: str = ""):
        self.private_attributes = {}
        self._name = name.lower()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name.lower()

    def private_attributes(self, attrs: Dict[str, Any]) -> None:
        """
        Set the private attributes of the participant. These attributes will
        only be available within the tasks performed by the participants and
        will be filtered out prior to the task's state being transfered.

        Args:
            attrs: dictionary of ATTRIBUTE_NAME (str) -> object that will be accessible
                   within the participant's task.

                   Example:
                   {'train_loader' : torch.utils.data.DataLoader(...)}

                   In any task performed by this participant performed within the flow,
                   this attribute could be referenced with self.train_loader
        """
        self.private_attributes = attrs


class Collaborator(Participant):
    """
    Defines a collaborator participant
    """

    def __init__(
        self,
        name: str = "",
        private_attributes_callable: Callable = None,
        num_cpus: int = 0,
        num_gpus: int = 0.0,
        **kwargs
    ):
        """
        Create collaborator object with custom resources and a callable
        function to assign private attributes

        Parameters:
        name (str): Name of the collaborator. default=""

        private_attributes_callable (Callable): A function which returns collaborator
        private attributes for each collaborator. In case private_attributes are not
        required this can be omitted. default=None

        num_cpus (int): Specifies how many cores to use for the collaborator step exection.
        This will only be used if backend is set to ray. default=0

        num_gpus (float): Specifies how many GPUs to use to accerlerate the collaborator
        step exection. This will only be used if backend is set to ray. default=0

        kwargs (dict): Parameters required to call private_attributes_callable function.
        The key of the dictionary must match the arguments to the private_attributes_callable.
        default={}
        """
        super().__init__(name=name)
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.kwargs = kwargs

        if private_attributes_callable is None:
            self.private_attributes_callable = private_attributes_callable
        else:
            if not callable(private_attributes_callable):
                raise Exception(
                    "private_attributes_callable  parameter must be a callable"
                )
            else:
                self.private_attributes_callable = private_attributes_callable

    def get_name(self) -> str:
        """Get collaborator name"""
        return self._name

    def initialize_private_attributes(self) -> None:
        """
        initialize private attributes of Collaborator object by invoking
        the callable specified by user
        """
        if self.private_attributes_callable is not None:
            self.private_attributes = self.private_attributes_callable(**self.kwargs)

    def __set_collaborator_attrs_to_clone(self, clone: Any) -> None:
        """
        Set collaborator private attributes to FLSpec clone before transitioning
        from Aggregator step to collaborator steps
        """
        # set collaborator private attributes as
        # clone attributes
        for name, attr in self.private_attributes.items():
            setattr(clone, name, attr)

    def __delete_collab_attrs_from_clone(self, clone: Any) -> None:
        """
        Remove collaborator private attributes from FLSpec clone before
        transitioning from Collaborator step to Aggregator step
        """
        # Update collaborator private attributes by taking latest
        # parameters from clone, then delete attributes from clone.
        for attr_name in self.private_attributes:
            if hasattr(clone, attr_name):
                self.private_attributes.update({attr_name: getattr(clone, attr_name)})
                delattr(clone, attr_name)

    def execute_func(self, ctx: Any, f_name: str, callback: Callable) -> Any:
        """
        Execute remote function f
        """
        self.__set_collaborator_attrs_to_clone(ctx)

        callback(ctx, f_name)

        self.__delete_collab_attrs_from_clone(ctx)

        return ctx


class Aggregator(Participant):
    """
    Defines an aggregator participant
    """

    def __init__(
        self,
        name: str = "",
        private_attributes_callable: Callable = None,
        num_cpus: int = 0,
        num_gpus: int = 0.0,
        **kwargs
    ):
        """
        Create aggregator object with custom resources and a callable
        function to assign private attributes

        Parameters:
        name (str): Name of the aggregator. default=""

        private_attributes_callable (Callable): A function which returns aggregator
        private attributes. In case private_attributes are not required this can be omitted.
        default=None

        num_cpus (int): Specifies how many cores to use for the aggregator step exection.
        This will only be used if backend is set to ray. default=0

        num_gpus (float): Specifies how many GPUs to use to accerlerate the aggregator
        step exection. This will only be used if backend is set to ray. default=0

        kwargs (dict): Parameters required to call private_attributes_callable function.
        The key of the dictionary must match the arguments to the private_attributes_callable.
        default={}
        """
        super().__init__(name=name)
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.kwargs = kwargs

        if private_attributes_callable is None:
            self.private_attributes_callable = private_attributes_callable
        else:
            if not callable(private_attributes_callable):
                raise Exception(
                    "private_attributes_callable parameter must be a callable"
                )
            else:
                self.private_attributes_callable = private_attributes_callable

    def get_name(self) -> str:
        """Get aggregator name"""
        return self.name

    def initialize_private_attributes(self) -> None:
        """
        initialize private attributes of Aggregator object by invoking
        the callable specified by user
        """
        if self.private_attributes_callable is not None:
            self.private_attributes = self.private_attributes_callable(**self.kwargs)

    def __set_agg_attrs_to_clone(self, clone: Any) -> None:
        """
        Set aggregator private attributes to FLSpec clone before transition
        from Aggregator step to collaborator steps
        """
        # set aggregator private attributes as
        # clone attributes
        for name, attr in self.private_attributes.items():
            setattr(clone, name, attr)

    def __delete_agg_attrs_from_clone(self, clone: Any) -> None:
        """
        Remove aggregator private attributes from FLSpec clone before
        transition from Aggregator step to collaborator steps
        """
        # Update aggregator private attributes by taking latest
        # parameters from clone, then delete attributes from clone.
        for attr_name in self.private_attributes:
            if hasattr(clone, attr_name):
                self.private_attributes.update({attr_name: getattr(clone, attr_name)})
                delattr(clone, attr_name)

    def execute_func(self, ctx: Any, f_name: str, callback: Callable,
                     clones: Optional[Any] = None) -> Any:
        """
        Execute remote function f
        """
        self.__set_agg_attrs_to_clone(ctx)

        if clones is not None:
            callback(ctx, f_name, clones)
        else:
            callback(ctx, f_name)

        self.__delete_agg_attrs_from_clone(ctx)

        return ctx
