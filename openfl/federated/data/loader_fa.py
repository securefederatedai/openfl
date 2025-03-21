# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""DataLoader module."""


class FederatedAnalyticsDataLoader:
    """A base class used to represent a Federated Learning Data Loader.

    This class should be inherited by any data loader class specific to a
    machine learning framework.

    Attributes:
        None
    """

    def __init__(self, **kwargs):
        """Initializes the FA DataLoader object.

        Args:
            kwargs: Additional arguments to pass to the function.
        """
        pass

    def query(self, **kwargs):
        """
        Query the data loader with specific parameters.
        This method should be implemented by subclasses to provide
        functionality for querying data based on the provided keyword arguments.
        Args:
            **kwargs: Arbitrary keyword arguments that specify the query parameters.
        Raises:
            NotImplementedError: This method is not implemented and should be
                                 overridden by subclasses.
        """
        raise NotImplementedError
