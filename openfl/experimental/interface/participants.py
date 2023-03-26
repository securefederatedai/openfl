# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.interface.participants module."""

import importlib
from typing import Dict
from typing import Any


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

__shard_descriptor = None

class Collaborator(Participant):
    """
    Defines a collaborator participant
    """
    def __init__(self, shard_descriptor_path, **kwargs):
        super().__init__(**kwargs)

        __shard_descriptor = globals().get("__shard_descriptor", None)

        if __shard_descriptor is None:
            ShardDescriptor = importlib.import_module(shard_descriptor_path).ShardDescriptor
            __shard_descriptor = ShardDescriptor()
            globals()['__shard_descriptor'] = __shard_descriptor

        self.__assign_private_attr()


    def __assign_private_attr(self):
        __shard_descriptor = globals().get("__shard_descriptor", None)

        self.private_attributes = __shard_descriptor.get()


class Aggregator(Participant):
    """
    Defines an aggregator participant
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
