# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import importlib

if importlib.util.find_spec('tensorflow'):
    from openfl.utilities.optimizers.keras.fedprox import FedProxOptimizer # NOQA
