# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def collaborator_private_attrs(index):
    return {
        "train_loader": np.random.rand(index * 50, 28, 28),
        "test_loader": np.random.rand(index * 10, 28, 28),
    }
