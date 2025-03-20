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

    def get_data(self):
        raise NotImplementedError
