# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""openfl.experimental.utilities.resources module."""

from logging import getLogger
from subprocess import PIPE, run

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
        logger.warning(
            f'No GPUs found! If this is a mistake please try running "{command}" ' + "manually."
        )
        return 0
