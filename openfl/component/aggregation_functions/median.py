# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Median module."""

import numpy as np


def median(tensors, *_):
    """Compute median."""
    return np.median(tensors, axis=0)
