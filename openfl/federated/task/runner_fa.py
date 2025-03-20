# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
Base classes for Federated Analytics.

You may copy this file as the starting point of your own keras model.
"""

from logging import getLogger


class FederatedAnalyticsTaskRunner:
    """The base class for Federated Analytics."""

    def __init__(self, data_loader, **kwargs):
        """Intializes the TaskRunner object.

        Args:
            data_loader: The data_loader object
            **kwargs: Additional parameters to pass to the function.
        """
        self.data_loader = data_loader
        self.set_logger()

    def set_logger(self):
        """Set up the log object.

        Returns:
            None
        """
        self.logger = getLogger(__name__)

    def analysis(self, **kwargs):
        raise NotImplementedError

    def get_query_data_size(self):
        return self.data_loader.get_query_data_size()
