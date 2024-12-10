# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Logs utilities."""

import json
import logging
import os

import psutil
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


def get_memory_usage() -> dict:
    """Return memory usage details of the current process.

    Returns:
        dict: A dictionary containing memory usage details.
    """
    process = psutil.Process(os.getpid())
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    memory_usage = {
        "process_memory": round(process.memory_info().rss / (1024**2), 2),
        "virtual_memory": {
            "total": round(virtual_memory.total / (1024**2), 2),
            "available": round(virtual_memory.available / (1024**2), 2),
            "percent": virtual_memory.percent,
            "used": round(virtual_memory.used / (1024**2), 2),
            "free": round(virtual_memory.free / (1024**2), 2),
            "active": round(virtual_memory.active / (1024**2), 2),
            "inactive": round(virtual_memory.inactive / (1024**2), 2),
            "buffers": round(virtual_memory.buffers / (1024**2), 2),
            "cached": round(virtual_memory.cached / (1024**2), 2),
            "shared": round(virtual_memory.shared / (1024**2), 2),
        },
        "swap_memory": {
            "total": round(swap_memory.total / (1024**2), 2),
            "used": round(swap_memory.used / (1024**2), 2),
            "free": round(swap_memory.free / (1024**2), 2),
            "percent": swap_memory.percent,
        },
    }
    return memory_usage


def write_memory_usage_to_file(memory_usage_dict, file_name):
    """
    Write memory usage details to a file.

    Args:
        memory_usage_dict (dict): The memory usage details to write.
        file_name (str): The name of the file to write to.

    Returns:
        None
    """
    file_path = os.path.join("logs", file_name)
    with open(file_path, "w") as f:
        json.dump(memory_usage_dict, f, indent=4)
