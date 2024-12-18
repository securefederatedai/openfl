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

# Docker specific constants
CREATE_OPENFL_NW = "docker network create"
REMOVE_OPENFL_NW = "docker network rm"
DOCKER_NETWORK_NAME = "openfl"
DEFAULT_OPENFL_IMAGE = "openfl:latest"

AGG_WORKSPACE_PATH = "{}/aggregator/workspace" # example - /tmp/my_federation/aggregator/workspace
COL_WORKSPACE_PATH = "{}/{}/workspace"  # example - /tmp/my_federation/collaborator1/workspace
AGG_PLAN_PATH = "{}/aggregator/workspace/plan"  # example - /tmp/my_federation/aggregator/workspace/plan
COL_PLAN_PATH = "{}/{}/workspace/plan"  # example - /tmp/my_federation/collaborator1/workspace/plan

AGG_COL_RESULT_FILE = "{0}/{1}/workspace/{1}.log"  # example - /tmp/my_federation/aggregator/workspace/aggregator.log

AGG_WORKSPACE_ZIP_NAME = "workspace.zip"

# Memory logs related
AGG_MEM_USAGE_JSON = "{}/aggregator/workspace/logs/aggregator_memory_usage.json"  # example - /tmp/my_federation/aggregator/workspace/logs/aggregator_memory_usage.json
COL_MEM_USAGE_JSON = "{0}/{1}/workspace/logs/{1}_memory_usage.json"  # example - /tmp/my_federation/collaborator1/workspace/logs/collaborator1_memory_usage.json
