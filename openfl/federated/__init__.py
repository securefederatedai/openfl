# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.federated package."""

import importlib
from openfl.federated.plan import Plan  # NOQA
from openfl.federated.task import TaskRunner  # NOQA
from openfl.federated.data import DataLoader  # NOQA

if importlib.util.find_spec('tensorflow'):
    from openfl.federated.task import TensorFlowTaskRunner, KerasTaskRunner, FederatedModel  # NOQA
    from openfl.federated.data import TensorFlowDataLoader, KerasDataLoader, FederatedDataSet  # NOQA
if importlib.util.find_spec('torch'):
    from openfl.federated.task import PyTorchTaskRunner, FederatedModel  # NOQA
    from openfl.federated.data import PyTorchDataLoader, FederatedDataSet  # NOQA
