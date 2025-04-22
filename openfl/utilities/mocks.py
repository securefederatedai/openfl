# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Mock objects to eliminate extraneous dependencies"""


class MockDataLoader:
    """Placeholder dataloader for when data is not available"""

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_feature_shape(self):
        return self.input_shape

    def get_train_data_size(self):
        return 0

    def get_valid_data_size(self):
        return 0
