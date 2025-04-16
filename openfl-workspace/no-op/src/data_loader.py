# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.federated.data.loader import DataLoader

class NoOpDataLoader(DataLoader):
    """No-op data loader.

    This class is used when no data loader is needed.
    """

    def __init__(self, **kwargs):
        """Initializes the NoOpDataLoader object.

        Args:
            kwargs: Additional arguments to pass to the function.
        """
        super().__init__(**kwargs)

    def get_feature_shape(self):
        return None

    def get_train_loader(self, **kwargs):
        return None

    def get_valid_loader(self):
        return None

    def get_infer_loader(self):
        return None

    def get_train_data_size(self):
        return 0

    def get_valid_data_size(self):
        return 0

    def get_infer_data_size(self):
        return 0

    def get_train_data(self):
        return None

    def get_valid_data(self):
        return None

    def get_infer_data(self):
        return None
