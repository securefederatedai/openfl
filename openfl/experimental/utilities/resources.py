# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""openfl.experimental.utilities.resources module."""

from logging import getLogger
from subprocess import run, PIPE

logger = getLogger(__name__)


def get_number_of_gpus() -> int:
    """
    Returns number of NVIDIA GPUs attached to the machine.

    Args:
        None
    Returns:
        int: Number of NVIDIA GPUs
    """
    # Execute the nvidia-smi command.
    command = "nvidia-smi --list-gpus"
    try:
        op = run(command.strip().split(), shell=False, stdout=PIPE, stderr=PIPE)
        stdout = op.stdout.decode().strip()
        return len(stdout.split("\n"))
    except FileNotFoundError:
        logger.warning(f'No GPUs found! If this is a mistake please try running "{command}" '
                       + 'manually.')
        return 0
