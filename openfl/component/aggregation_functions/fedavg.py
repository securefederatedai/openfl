# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"Federated averaging module."

import numpy as np


def weighted_average(tensors, weights):
    """Compute simple weighted average (FedAvg operation)."""
    return np.average(tensors, weights=weights, axis=0)
