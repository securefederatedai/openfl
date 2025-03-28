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
    # Create a logger instance
    logger = logging.getLogger()

    # Add a custom log level for METRIC
    metric = 25
    add_log_level("METRIC", metric)

    if isinstance(log_level, str):
        log_level = log_level.upper()

    # Set the log level for the logger
    logger.setLevel(log_level)

    console = Console(width=160)
    console_handler = RichHandler(console=console)

    # Console handler includes date and log level, do not add it again
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # Create a file handler if log_file is provided
    if log_file:
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
