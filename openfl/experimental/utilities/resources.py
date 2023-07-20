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
    command = "nvidia-smi --query-gpu=name --format=csv,noheader"
    op = run(command.strip().split(), shell=False, stdout=PIPE, stderr=PIPE)

    # If returncode is 0 then command ran successfully
    if op.returncode is 0:
        stdout = op.stdout.decode().strip()

        return len(stdout.split("\n"))
    else:
        stderr = op.stderr.decode().strip()
        logger.warning(f"No GPUs found! If this is a mistake please check nvidia-smi error {stderr}")
        return 0
