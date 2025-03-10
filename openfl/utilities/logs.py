# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Logs utilities."""

import logging

from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger()


def setup_loggers(log_level=logging.INFO, log_file=None):
    """Configure loggers.

    This function sets up the root logger to log messages with a certain
    minimum level and a specific format.

    Args:
        log_level (int, optional): The minimum level of messages to log.
            Defaults to logging.INFO.
        log_file (str, optional): The file to which log messages should be written.
    """
    root = logging.getLogger()
    root.setLevel(log_level)
    console = Console(width=160, force_terminal=True)
    handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        console=console,
    )
    # formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
