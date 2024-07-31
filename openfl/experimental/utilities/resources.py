# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.utilities.resources module."""

from logging import getLogger
from subprocess import PIPE, run

logger = getLogger(__name__)


def get_number_of_gpus() -> int:
    """Returns number of NVIDIA GPUs attached to the machine.

    This function executes the `nvidia-smi --list-gpus` command to get the
    list of GPUs.
    If the command fails (e.g., NVIDIA drivers are not installed), it logs a
    warning and returns 0.

    Returns:
        int: The number of NVIDIA GPUs attached to the machine.
    """
    # Execute the nvidia-smi command.
    command = "nvidia-smi --list-gpus"
    try:
        op = run(command.strip().split(), shell=False, stdout=PIPE, stderr=PIPE)
        stdout = op.stdout.decode().strip()
        return len(stdout.split("\n"))
    except FileNotFoundError:
        logger.warning(
            f'No GPUs found! If this is a mistake please try running "{command}" ' + "manually."
        )
        return 0
