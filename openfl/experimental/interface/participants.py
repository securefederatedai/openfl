# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.interface.participants module."""

from typing import Dict
from typing import Any


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

    def private_attributes(self, attrs: Dict[str, Any]) -> None:
        """Set the private attributes of the participant. These attributes will
        only be available within the tasks performed by the participants and
        will be filtered out prior to the task's state being transfered.

        Args:
            attrs (Dict[str, Any]): dictionary of ATTRIBUTE_NAME (str) -> object that will be accessible
                   within the participant's task.

                   Example:
                   {'train_loader' : torch.utils.data.DataLoader(...)}

                   In any task performed by this participant performed within the flow,
                   this attribute could be referenced with self.train_loader
        """
        self.private_attributes = attrs


class Collaborator(Participant):
    """Class for a collaborator participant, derived from the Participant class."""

    def __init__(self, **kwargs):
        """Initializes the Collaborator object with variable length arguments.

        Args:
            **kwargs: Variable length argument list.
        """
        super().__init__(**kwargs)


class Aggregator(Participant):
    """Class for an aggregator participant, derived from the Participant class."""

    def __init__(self, **kwargs):
        """Initializes the Aggregator object with variable length arguments.

        Args:
            **kwargs: Variable length argument list.
        """
        super().__init__(**kwargs)
