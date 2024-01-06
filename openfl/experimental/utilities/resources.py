# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.utilities.resources module."""

from torch.cuda import device_count


def get_number_of_gpus():
    """Gets the number of GPUs available on the current device.

    Returns:
        device_count (int): The number of GPUs available on the current device.

    .. note::
        This function currently depends on PyTorch. Future work includes removing this dependency.
    """
    # TODO remove pytorch dependency
    return device_count()
