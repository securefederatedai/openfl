# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.utilities.resources module."""

from torch.cuda import device_count


def get_number_of_gpus():
    # TODO remove pytorch dependency
    return device_count()
