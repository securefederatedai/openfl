# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torch.cuda import device_count


def get_number_of_gpus() -> int:
    return device_count()
