# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.interface.participants module."""

import yaml
import importlib
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
    def __init__(self, config_filename, **kwargs):
        super().__init__(**kwargs)

        filepath, class_name = self.read_config_file(config_filename).split(".")

        shard_descriptor_module = importlib.import_module(filepath)
        ShardDescriptor = getattr(shard_descriptor_module, class_name)

        self.__shard_descriptor = ShardDescriptor(config_filename)

        self.assign_private_attributes()


    def read_config_file(self, config_filename):
        with open(config_filename, "r") as f:
            config = yaml.load(f, Loader=SafeLoader)

        return config["shard_descriptor"]["template"]


    def assign_private_attributes(self):
        self.private_attributes = self.__shard_descriptor.get()
        del self.__shard_descriptor


class Aggregator(Participant):
    """
    Defines an aggregator participant
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
