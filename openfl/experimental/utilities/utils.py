# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torch.cuda import device_count


def get_number_of_gpus() -> int:
    """Gets the number of GPUs available.

    Returns:
        int: The number of GPUs available.
    """
    return device_count()
