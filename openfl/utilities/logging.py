# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Logs utilities."""

import logging

from rich.console import Console
from rich.logging import RichHandler

from openfl.utilities import add_log_level


def setup_logger(log_level=logging.INFO, log_file=None):
    """Configure loggers.

    This function sets up the root logger to log messages with a certain
    minimum level and a specific format.

    Args:
        log_level (int, optional): The minimum level of messages to log.
            Defaults to logging.INFO.
        log_file (str, optional): The file to which log messages should be written.
    """
    metric = 25
    add_log_level("METRIC", metric)

    if isinstance(log_level, str):
        log_level = log_level.upper()

    root = logging.getLogger()
    root.setLevel(log_level)

    formatter = logging.Formatter("%(message)s")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    console = Console(width=160, force_terminal=True)
    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        console=console,
    )
    rich_handler.setFormatter(formatter)
    root.addHandler(rich_handler)
