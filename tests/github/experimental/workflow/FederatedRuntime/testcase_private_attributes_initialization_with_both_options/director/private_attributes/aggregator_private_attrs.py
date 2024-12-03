# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def aggregator_private_attrs():
    return {"test_loader_agg_via_callable": np.random.rand(10, 28, 28)}  # Random data


aggregator_private_attributes = {"test_loader_agg": np.random.rand(10, 28, 28)}  # Random data
