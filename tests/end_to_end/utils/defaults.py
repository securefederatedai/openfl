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
    TORCH_HISTOLOGY_S3 = "torch/histology_s3"
    TORCH_MNIST = "torch/mnist"
    TORCH_MNIST_EDEN_COMPRESSION = "torch/mnist_eden_compression"
    TORCH_MNIST_STRAGGLER_CHECK = "torch/mnist_straggler_check"
    KERAS_TENSORFLOW_MNIST = "keras/tensorflow/mnist"
    XGB_HIGGS = "xgb_higgs"
    GANDLF_SEG_TEST = "gandlf_seg_test"
    FLOWER_APP_PYTORCH = "flower-app-pytorch"
    NO_OP = "no-op"
    FEDERATED_ANALYTICS_HISTOGRAM = "federated_analytics/histogram"
    FEDERATED_ANALYTICS_SMOKERS_HEALTH = "federated_analytics/smokers_health"

NUM_COLLABORATORS = 2
NUM_ROUNDS = 5
WORKSPACE_NAME = "my_federation"
SUCCESS_MARKER = "✔️ OK"

# Docker specific defaults
CREATE_OPENFL_NW = "docker network create"
REMOVE_OPENFL_NW = "docker network rm"
DOCKER_NETWORK_NAME = "openfl"
DEFAULT_OPENFL_IMAGE = "openfl:latest"
DEFAULT_OPENFL_DOCKERFILE = "openfl-docker/Dockerfile.base"

AGG_WORKSPACE_PATH = "{}/aggregator/workspace" # example - /tmp/my_federation/aggregator/workspace
COL_WORKSPACE_PATH = "{}/{}/workspace"  # example - /tmp/my_federation/collaborator1/workspace
AGG_PLAN_PATH = "{}/aggregator/workspace/plan"  # example - /tmp/my_federation/aggregator/workspace/plan
COL_PLAN_PATH = "{}/{}/workspace/plan"  # example - /tmp/my_federation/collaborator1/workspace/plan

COL_DATA_FILE = "{}/{}/workspace/plan/data.yaml"  # example - /tmp/my_federation/collaborator1/workspace/plan/data.yaml

DATA_SETUP_FILE = "setup_data.py" # currently xgb_higgs is using this file to setup data

AGG_COL_RESULT_FILE = "{0}/{1}/workspace/{1}.log"  # example - /tmp/my_federation/aggregator/workspace/aggregator.log

DFLT_WORKSPACE_NAME = "workspace"

# Memory logs related
AGG_MEM_USAGE_LOGFILE = "{}/aggregator/workspace/logs/aggregator_memory_usage.txt"  # example - /tmp/my_federation/aggregator/workspace/logs/aggregator_memory_usage.txt
COL_MEM_USAGE_LOGFILE = "{0}/{1}/workspace/logs/{1}_memory_usage.txt"  # example - /tmp/my_federation/collaborator1/workspace/logs/collaborator1_memory_usage.txt

AGG_START_CMD = "fx aggregator start"
AGG_END_MSG = "Experiment Completed"
COL_START_CMD = "fx collaborator start -n {}"
COL_END_MSG = "Received shutdown signal"

COL_CERTIFY_CMD = "fx collaborator certify --import 'agg_to_col_{}_signed_cert.zip'"
EXCEPTION = "Exception"
AGG_METRIC_MODEL_ACCURACY_KEY = "aggregator/aggregated_model_validation/accuracy"
COL_TLS_END_MSG = "TLS connection established."


class TransportProtocol(Enum):
    """
    Enum class to define the transport protocol.
    """
    GRPC = "grpc"
    REST = "rest"

AGGREGATOR_REST_CLIENT = "Starting Aggregator REST Server"
AGGREGATOR_gRPC_CLIENT = "Starting Aggregator gRPC Server"

# For S3 and MinIO
MINIO_ROOT_USER = "minioadmin"
MINIO_ROOT_PASSWORD = "minioadmin"
MINIO_HOST = "localhost"
MINIO_PORT = 9000
MINIO_CONSOLE_PORT = 9001
MINIO_URL = f"http://{MINIO_HOST}:{MINIO_PORT}"
MINIO_CONSOLE_URL = f"http://{MINIO_HOST}:{MINIO_CONSOLE_PORT}"
MINIO_DATA_FOLDER = "minio_data"

# For Azure Blob Storage
AZURE_STORAGE_HOST = "localhost"
AZURE_STORAGE_PORT = 10000
AZURE_STORAGE_ENDPOINTS_PROTOCOL = "http"
AZURE_STORAGE_ACCOUNT_NAME = "devstoreaccount1"
# IMP: The account key is provided by Azure for local development storage
# and is not a real key. It is used for testing purposes only.
AZURE_STORAGE_ACCOUNT_KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
AZURE_BLOB_ENDPOINT = f"{AZURE_STORAGE_ENDPOINTS_PROTOCOL}://{AZURE_STORAGE_HOST}:{AZURE_STORAGE_PORT}/{AZURE_STORAGE_ACCOUNT_NAME}"
