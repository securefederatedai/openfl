# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib

if importlib.util.find_spec("tensorflow") is not None:
    from openfl.utilities.optimizers.keras.fedprox import FedProxOptimizer  # NOQA
