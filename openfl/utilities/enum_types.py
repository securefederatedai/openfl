# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enum types for device policy and opt treatment."""

from enum import Enum


class DevicePolicy(Enum):
    """Device assignment policy."""

    CPU_ONLY = 1
    CUDA_PREFERRED = 2


class OptTreatment(Enum):
    """Optimizer Methods.

    - RESET tells each collaborator to reset the optimizer state at the beginning
    of each round.

    - CONTINUE_LOCAL tells each collaborator to continue with the local optimizer
    state from the previous round.

    - CONTINUE_GLOBAL tells each collaborator to continue with the federally
    averaged optimizer state from the previous round.
    """

    RESET = 1
    CONTINUE_LOCAL = 2
    CONTINUE_GLOBAL = 3
