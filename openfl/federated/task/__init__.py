# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Task package."""
import os
from importlib import util
from warnings import catch_warnings, simplefilter

with catch_warnings():
    simplefilter(action="ignore", category=FutureWarning)
    if util.find_spec("tensorflow") is not None:
        # ignore deprecation warnings in command-line interface
        import tensorflow  # NOQA

from openfl.federated.task.runner import TaskRunner  # NOQA

print("inside init inside task")
print("util.find_spec(torch)", util.find_spec("torch"))
print("util.find_spec(keras)", util.find_spec("keras"))
print("util.find_spec(jax)", util.find_spec("jax"))
if util.find_spec("keras") is not None:
    if util.find_spec("torch") is not None:
        # This guide can only be run with the torch backend.
        os.environ["KERAS_BACKEND"] = "torch"
    elif util.find_spec("tensorflow") is not None:
        # This guide can only be run with the torch backend.
        os.environ["KERAS_BACKEND"] = "tensorflow"
    elif util.find_spec("jax") is not None:
        # This guide can only be run with the torch backend.
        os.environ["KERAS_BACKEND"] = "jax"
    print(os.environ["KERAS_BACKEND"])
    from openfl.federated.task.fl_model import FederatedModel  # NOQA
    from openfl.federated.task.runner_keras import KerasTaskRunner  # NOQA
if util.find_spec("torch") is not None:
    from openfl.federated.task.fl_model import FederatedModel  # NOQA
    from openfl.federated.task.runner_pt import PyTorchTaskRunner  # NOQA
if util.find_spec("xgboost") is not None:
    from openfl.federated.task.fl_model import FederatedModel  # NOQA
    from openfl.federated.task.runner_xgb import XGBoostTaskRunner  # NOQA
