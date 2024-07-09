# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""openfl.federated package."""

import importlib

from openfl.federated.data import DataLoader  # NOQA
from openfl.federated.plan import Plan  # NOQA
from openfl.federated.task import TaskRunner  # NOQA

if importlib.util.find_spec('tensorflow') is not None:
    from openfl.federated.data import FederatedDataSet  # NOQA
    from openfl.federated.data import KerasDataLoader, TensorFlowDataLoader
    from openfl.federated.task import FederatedModel  # NOQA
    from openfl.federated.task import KerasTaskRunner, TensorFlowTaskRunner
if importlib.util.find_spec('torch') is not None:
    from openfl.federated.data import FederatedDataSet  # NOQA
    from openfl.federated.data import PyTorchDataLoader
    from openfl.federated.task import FederatedModel  # NOQA
    from openfl.federated.task import PyTorchTaskRunner
