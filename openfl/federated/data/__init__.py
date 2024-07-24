# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Data package."""

import importlib
from warnings import catch_warnings, simplefilter

with catch_warnings():
    simplefilter(action="ignore", category=FutureWarning)
    if importlib.util.find_spec("tensorflow") is not None:
        # ignore deprecation warnings in command-line interface
        import tensorflow  # NOQA

from openfl.federated.data.loader import DataLoader  # NOQA

if importlib.util.find_spec("tensorflow") is not None:
    from openfl.federated.data.federated_data import FederatedDataSet  # NOQA
    from openfl.federated.data.loader_keras import KerasDataLoader  # NOQA
    from openfl.federated.data.loader_tf import TensorFlowDataLoader  # NOQA

if importlib.util.find_spec("torch") is not None:
    from openfl.federated.data.federated_data import FederatedDataSet  # NOQA
    from openfl.federated.data.loader_pt import PyTorchDataLoader  # NOQA
