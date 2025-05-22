# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom GaNDLF DataLoader module."""

from openfl.federated.data.loader_gandlf import GaNDLFDataLoaderWrapper


class GaNDLFDataLoader(GaNDLFDataLoaderWrapper):
    """A custom data loader for the Generally Nuanced Deep Learning Framework (GaNDLF).

    This class extends the GaNDLFDataLoaderWrapper to provide a custom implementation
    of the get_feature_shape method.

    Attributes:
        Inherits all attributes from GaNDLFDataLoaderWrapper.
    """
    def __init__(self, data_path=None, feature_shape=None, **kwargs):
        """Initializes the GaNDLFDataLoader object.

        Args:
            data_path (str, optional): The path to the directory containing the data.
                If None, initialize for model creation only.
            feature_shape (tuple, optional): The shape of an example feature array.
                If None, will be derived from GANDLF config or default to [32, 32, 32].
            **kwargs: Additional arguments to pass to the function.
        """
        super().__init__(data_path=data_path, **kwargs)
        self.feature_shape = [32, 32, 32]

    def get_feature_shape(self):
        """Returns the shape of an example feature array.

        This method overrides the parent class method to provide a fixed feature shape
        instead of relying on the GaNDLF configuration.

        Returns:
            list: The shape of an example feature array.
        """
        # Define a fixed feature shape for this specific application
        # Use standard 3D patch size for medical imaging segmentation
        return self.feature_shape
