# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Logs utilities."""

import logging

from rich.console import Console
from rich.logging import RichHandler


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
