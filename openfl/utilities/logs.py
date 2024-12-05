# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Logs utilities."""

import logging

from rich.console import Console
from rich.logging import RichHandler
from tensorboardX import SummaryWriter

writer = None


def get_writer():
    """Create global writer object.

    This function creates a global `SummaryWriter` object for logging to
    TensorBoard.
    """
    global writer
    if not writer:
        writer = SummaryWriter("./logs/tensorboard", flush_secs=5)


def write_metric(node_name, task_name, metric_name, metric, round_number):
    """Write metric callback.

    This function logs a metric to TensorBoard.

    Args:
        node_name (str): The name of the node.
        task_name (str): The name of the task.
        metric_name (str): The name of the metric.
        metric (float): The value of the metric.
        round_number (int): The current round number.
    """
    get_writer()
    writer.add_scalar(f"{node_name}/{task_name}/{metric_name}", metric, round_number)


def setup_loggers(log_level=logging.INFO):
    """Configure loggers.

    This function sets up the root logger to log messages with a certain
    minimum level and a specific format.

    Args:
        log_level (int, optional): The minimum level of messages to log.
            Defaults to logging.INFO.
    """
    root = logging.getLogger()
    root.setLevel(log_level)
    console = Console(width=160)
    handler = RichHandler(console=console)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
