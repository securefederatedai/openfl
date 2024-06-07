# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.federated package."""

import importlib.util
from .plan import Plan  # NOQA
from .task import TaskRunner  # NOQA
from .data import DataLoader  # NOQA

if importlib.util.find_spec('tensorflow'):
    from .task import TensorFlowTaskRunner, TensorFlowTaskRunnerV1, KerasTaskRunner, FederatedModel  # NOQA
    from .data import TensorFlowDataLoader, KerasDataLoader, FederatedDataSet  # NOQA
if importlib.util.find_spec('torch'):
    from .task import PyTorchTaskRunner, FederatedModel  # NOQA
    from .data import PyTorchDataLoader, FederatedDataSet  # NOQA
