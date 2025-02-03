# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.federated package."""

import os
from importlib import util

from openfl.federated.data import DataLoader  # NOQA
from openfl.federated.plan import Plan  # NOQA
from openfl.federated.task import TaskRunner  # NOQA
print("inside init")
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
    from openfl.federated.data import FederatedDataSet  # NOQA
    from openfl.federated.data import KerasDataLoader
    from openfl.federated.task import FederatedModel  # NOQA
    from openfl.federated.task import KerasTaskRunner
if util.find_spec("torch") is not None:
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    from openfl.federated.data import FederatedDataSet  # NOQA
    from openfl.federated.data import PyTorchDataLoader
    from openfl.federated.task import FederatedModel  # NOQA
    from openfl.federated.task import PyTorchTaskRunner
if util.find_spec("xgboost") is not None:
    from openfl.federated.data import FederatedDataSet  # NOQA
    from openfl.federated.data import XGBoostDataLoader
    from openfl.federated.task import FederatedModel  # NOQA
    from openfl.federated.task import XGBoostTaskRunner


__all__ = [
    "Plan",
    "TaskRunner",
    "DataLoader",
]
