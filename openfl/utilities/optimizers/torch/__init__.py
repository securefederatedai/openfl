# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""PyTorch optimizers package."""
import importlib

if importlib.util.find_spec("torch") is not None:
    from openfl.utilities.optimizers.torch.fedprox import FedProxAdam  # NOQA
    from openfl.utilities.optimizers.torch.fedprox import FedProxOptimizer  # NOQA
