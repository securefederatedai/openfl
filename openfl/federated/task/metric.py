"""Metric module."""

from typing import NamedTuple
import numpy as np


class Metric(NamedTuple):
    """Entity representing training results."""

    name: str
    value: np.ndarray
