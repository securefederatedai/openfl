# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Data package."""

from importlib import util

from openfl.federated.data.loader import DataLoader  # NOQA

if util.find_spec("keras") is not None:
    from openfl.federated.data.loader_keras import KerasDataLoader  # NOQA

if util.find_spec("torch") is not None:
    from openfl.federated.data.loader_pt import PyTorchDataLoader  # NOQA

if util.find_spec("xgboost") is not None:
    from openfl.federated.data.loader_xgb import XGBoostDataLoader  # NOQA
