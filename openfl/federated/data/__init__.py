# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Data package."""

from importlib import util
from warnings import catch_warnings, simplefilter

with catch_warnings():
    simplefilter(action="ignore", category=FutureWarning)
    if util.find_spec("tensorflow") is not None:
        # ignore deprecation warnings in command-line interface
        import tensorflow  # NOQA

from openfl.federated.data.loader import DataLoader  # NOQA

if util.find_spec("keras") is not None:
    from openfl.federated.data.federated_data import FederatedDataSet  # NOQA
    from openfl.federated.data.loader_keras import KerasDataLoader  # NOQA

if util.find_spec("torch") is not None:
    from openfl.federated.data.federated_data import FederatedDataSet  # NOQA
    from openfl.federated.data.loader_pt import PyTorchDataLoader  # NOQA

if util.find_spec("xgboost") is not None:
    from openfl.federated.data.federated_data import FederatedDataSet  # NOQA
    from openfl.federated.data.loader_xgb import XGBoostDataLoader  # NOQA
