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
    KERAS_JAX_MNIST = "keras/jax/mnist"
    KERAS_MNIST = "keras/mnist"
    KERAS_TORCH_MNIST = "keras/torch/mnist"
    TORCH_HISTOLOGY = "torch/histology"
    TORCH_MNIST = "torch/mnist"
    TORCH_MNIST_EDEN_COMPRESSION = "torch/mnist_eden_compression"
    TORCH_MNIST_STRAGGLER_CHECK = "torch/mnist_straggler_check"
    XGB_HIGGS = "xgb_higgs"


NUM_COLLABORATORS = 2
NUM_ROUNDS = 5
WORKSPACE_NAME = "my_federation"
DEFAULT_MODEL_NAME = "torch/mnist"
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

COL_DATA_FILE = "{}/{}/workspace/plan/data.yaml"  # example - /tmp/my_federation/collaborator1/workspace/plan/data.yaml

DATA_SETUP_FILE = "setup_data.py" # currently xgb_higgs is using this file to setup data

AGG_COL_RESULT_FILE = "{0}/{1}/workspace/{1}.log"  # example - /tmp/my_federation/aggregator/workspace/aggregator.log

AGG_WORKSPACE_ZIP_NAME = "workspace.zip"

# Memory logs related
AGG_MEM_USAGE_JSON = "{}/aggregator/workspace/logs/aggregator_memory_usage.json"  # example - /tmp/my_federation/aggregator/workspace/logs/aggregator_memory_usage.json
COL_MEM_USAGE_JSON = "{0}/{1}/workspace/logs/{1}_memory_usage.json"  # example - /tmp/my_federation/collaborator1/workspace/logs/collaborator1_memory_usage.json

AGG_START_CMD = "fx aggregator start"
COL_START_CMD = "fx collaborator start -n {}"

COL_CERTIFY_CMD = "fx collaborator certify --import 'agg_to_col_{}_signed_cert.zip'"
DFLT_DOCKERIZE_IMAGE_NAME = "workspace"
