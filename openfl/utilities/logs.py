import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_loggers(log_level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(log_level)
    console = Console(width=160)
    handler = RichHandler(console=console)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
