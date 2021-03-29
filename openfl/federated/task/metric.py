from typing import NamedTuple
import numpy as np


class Metric(NamedTuple):
    name: str
    value: np.ndarray
