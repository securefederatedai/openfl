# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from rich.console import Console
from rich.logging import RichHandler

# Get the logger instance configured in conftest.py
logger = logging.getLogger()


def configure_logging(log_file, log_level):
    """
    Configures logging for the application.

    This function sets up logging to a specified file and the console with the given log level.
    It formats the log messages to include the timestamp, logger name, log level, filename,
    function name, and the actual log message.

    Args:
        log_file (str): Path to the log file.
        log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

    Raises:
        OSError: If there is an issue with creating the log file handler.
    """
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    # Rich strips colors if it detects its not writing to a terminal
    # That includes logging during GitHub workflow runs
    # Thus force_terminal is set to True to ensure colors are displayed
    console = Console(width=160, force_terminal=True)
    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        console=console,
    )
    rich_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    logger.handlers = []

    logger.addHandler(handler)
    logger.addHandler(rich_handler)
