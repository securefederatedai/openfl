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

"""Task package."""

import importlib
from warnings import catch_warnings, simplefilter

with catch_warnings():
    simplefilter(action="ignore", category=FutureWarning)
    if importlib.util.find_spec("tensorflow") is not None:
        # ignore deprecation warnings in command-line interface
        import tensorflow  # NOQA

from openfl.federated.task.runner import TaskRunner  # NOQA

if importlib.util.find_spec("tensorflow") is not None:
    from openfl.federated.task.fl_model import FederatedModel  # NOQA
    from openfl.federated.task.runner_keras import KerasTaskRunner  # NOQA
    from openfl.federated.task.runner_tf import TensorFlowTaskRunner  # NOQA
if importlib.util.find_spec("torch") is not None:
    from openfl.federated.task.fl_model import FederatedModel  # NOQA
    from openfl.federated.task.runner_pt import PyTorchTaskRunner  # NOQA
