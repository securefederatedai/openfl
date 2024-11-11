# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

# Define the model names. This is a non exhaustive list of models that can be used in the tests
class ModelName(Enum):
    """
    Enum class to define the model names.
    """
    # IMP - The model name must be same (and in uppercase) as the model value.
    # This is used to identify the model in the tests.
    TORCH_CNN_MNIST = "torch_cnn_mnist"
    KERAS_CNN_MNIST = "keras_cnn_mnist"
    TORCH_CNN_HISTOLOGY = "torch_cnn_histology"

NUM_COLLABORATORS = 2
NUM_ROUNDS = 5
WORKSPACE_NAME = "my_federation"
DEFAULT_MODEL_NAME = "torch_cnn_mnist"
SUCCESS_MARKER = "✔️ OK"
