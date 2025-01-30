# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

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
    formatter = logging.Formatter(
        "\n%(asctime)s - %(levelname)s: [%(filename)s - %(funcName)s - %(lineno)d]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
