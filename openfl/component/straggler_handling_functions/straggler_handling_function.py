# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Straggler handling module."""

from abc import ABC, abstractmethod


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
            bool: True if it is time to end the round early, False otherwise.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError
