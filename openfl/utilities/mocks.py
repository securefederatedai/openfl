# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Mock objects to eliminate extraneous dependencies"""


class MockDataLoader:
    """Placeholder dataloader for when data is not available"""

    def __init__(self, feature_shape):
        self.feature_shape = feature_shape

    def get_feature_shape(self):
        return self.feature_shape

    def get_train_data_size(self):
        return 0

    def get_valid_data_size(self):
        return 0
