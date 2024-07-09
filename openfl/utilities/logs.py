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

"""Logs utilities."""

import logging

from rich.console import Console
from rich.logging import RichHandler
from tensorboardX import SummaryWriter

writer = None


def get_writer():
    """Create global writer object."""
    global writer
    if not writer:
        writer = SummaryWriter('./logs/tensorboard', flush_secs=5)


def write_metric(node_name, task_name, metric_name, metric, round_number):
    """Write metric callback."""
    get_writer()
    writer.add_scalar(f'{node_name}/{task_name}/{metric_name}', metric, round_number)


def setup_loggers(log_level=logging.INFO):
    """Configure loggers."""
    root = logging.getLogger()
    root.setLevel(log_level)
    console = Console(width=160)
    handler = RichHandler(console=console)
    formatter = logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
