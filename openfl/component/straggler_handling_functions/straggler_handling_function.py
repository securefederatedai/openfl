# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Straggler handling module."""

from abc import ABC
from abc import abstractmethod


# TODO: Rename this file to "straggler_handling_policy.py"
# TODO: Rename package to "straggler_handling_policies"
class StragglerHandlingPolicy(ABC):
    """Federated Learning straggler handling interface."""

    @abstractmethod
    def start_policy(self, **kwargs) -> None:
        """
        Start straggler handling policy for collaborator for a particular round.
        NOTE: Refer CutoffTimeBasedStragglerHandling for reference.

        Args:
            **kwargs

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def reset_policy_for_round(self) -> None:
        """
        Reset policy variable for the next round.

        Args:
            None

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def straggler_cutoff_check(
        self, num_collaborators_done: int, num_all_collaborators: int, **kwargs
    ) -> bool:
        """
        Determines whether it is time to end the round early.

        Args:
            num_collaborators_done: int
                Number of collaborators finished.
            num_all_collaborators: int
                Total number of collaborators.

        Returns:
            bool
        """
        raise NotImplementedError
