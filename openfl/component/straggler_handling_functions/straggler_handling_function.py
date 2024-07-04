# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Straggler handling module."""

from abc import ABC, abstractmethod


class StragglerHandlingFunction(ABC):
    """Federated Learning straggler handling interface."""

    @abstractmethod
    def straggler_cutoff_check(self, **kwargs):
        """Determines whether it is time to end the round early.

        Args:
            **kwargs: Variable length argument list.

        Returns:
            bool: True if it is time to end the round early, False otherwise.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError
