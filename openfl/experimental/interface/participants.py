# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.interface.participants module."""

import yaml
import importlib
from copy import deepcopy
from typing import Dict
from typing import Any
from yaml.loader import SafeLoader

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
    def __init__(self, collab_config, **kwargs):
        super().__init__(**kwargs)
        self.__collaborator_config_file = collab_config


    def initialize_private_attributes(self):
        with open(self.__collaborator_config_file, "r") as f:
            config = yaml.load(f, Loader=SafeLoader)

        filepath, class_name = config["shard_descriptor"]["template"].split(".")
    
        shard_descriptor_module = importlib.import_module(filepath)
        ShardDescriptor = getattr(shard_descriptor_module, class_name)

        shard_descriptor = ShardDescriptor(self.__collaborator_config_file)

        self.private_attributes = deepcopy(shard_descriptor.get())


class Aggregator(Participant):
    """
    Defines an aggregator participant
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
