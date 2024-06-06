# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Task package."""

import importlib.util
from warnings import catch_warnings
from warnings import simplefilter

with catch_warnings():
    simplefilter(action='ignore', category=FutureWarning)
    if importlib.util.find_spec('tensorflow'):
        # ignore deprecation warnings in command-line interface
        import tensorflow  # NOQA

from .runner import TaskRunner  # NOQA

if importlib.util.find_spec('tensorflow'):
    from .runner_tf import TensorFlowTaskRunner, TensorFlowTaskRunner_v1  # NOQA
    from .runner_keras import KerasTaskRunner  # NOQA
    from .fl_model import FederatedModel  # NOQA
if importlib.util.find_spec('torch'):
    from .runner_pt import PyTorchTaskRunner  # NOQA
    from .fl_model import FederatedModel  # NOQA
