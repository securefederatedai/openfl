# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Task package."""

from warnings import catch_warnings, simplefilter
import pkgutil

with catch_warnings():
    simplefilter(action='ignore', category=FutureWarning)
    if pkgutil.find_loader('tensorflow'):
        # ignore deprecation warnings in command-line interface
        import tensorflow  # NOQA

from .runner import TaskRunner  # NOQA


if pkgutil.find_loader('tensorflow'):
    from .runner_tf import TensorFlowTaskRunner  # NOQA
    from .runner_keras import KerasTaskRunner  # NOQA
    from .fl_model import FederatedModel  # NOQA
if pkgutil.find_loader('torch'):
    from .runner_pt import PyTorchTaskRunner  # NOQA
    from .fl_model import FederatedModel  # NOQA
if pkgutil.find_loader('torch') and pkgutil.find_loader('tensorflow'):
    from .runner_fe import FastEstimatorTaskRunner  # NOQA
