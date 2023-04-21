# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.interface.participants module."""

import yaml
import importlib
from yaml.loader import SafeLoader
from typing import Dict, Any, Callable


class Participant:
    def __init__(self, name: str = ""):
        self.private_attributes = {}
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

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
    def __init__(self, **kwargs):
        """
        Create collaborator object
        """
        super().__init__(**kwargs)

    def get_name(self) -> str:
        """Get collaborator name"""
        return self._name

    def __set_collaborator_attrs_to_clone(self, clone: Any) -> None:
        """Set collaborator private attributes to clone"""
        # set collaborator private attributes as
        # clone attributes
        for name, attr in self.private_attributes.items():
            setattr(clone, name, attr)

    def __delete_collab_attrs_from_clone(self, clone: Any) -> None:
        """Delete collaborator private attributes from clone"""
        # if clone has collaborator private attributes
        # then remove it
        for attr_name in self.private_attributes:
            if hasattr(clone, attr_name):
                self.private_attributes.update(
                    {attr_name: getattr(clone, attr_name)}
                )
                delattr(clone, attr_name)

    def initialize_private_attributes(self) -> None:
        """
        Initialize private attributes to aggregator object
        """
        get_dataset = self.private_attributes.get("get_dataset", None)
        collaborator_index = self.private_attributes.get("index")
        if get_dataset is not None:
            self.private_attributes.update(get_dataset(collaborator_index))

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_private_attributes(self) -> None:
        """
        Assign private attributes to aggregator object
        """
        get_dataset = self.private_attributes.get("get_dataset", None)
        if get_dataset is not None:
            self.private_attributes.update(get_dataset())
