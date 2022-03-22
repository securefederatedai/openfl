# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Keras optimizers package."""
import pkgutil

if pkgutil.find_loader('tensorflow'):
    from .fedprox import FedProxOptimizer # NOQA
