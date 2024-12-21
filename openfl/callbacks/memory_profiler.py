# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os

import psutil

from openfl.callbacks.callback import Callback

logger = logging.getLogger(__name__)


class MemoryProfiler(Callback):
    """Profile memory usage of the current process at the end of each round.

    Attributes:
        log_dir: If set, writes logs as lines of JSON.
    """

    def __init__(self, log_dir: str = "./logs/"):
        super().__init__()
        self.log_dir = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir

    def on_round_end(self, round_num: int, logs=None):
        origin = self.params["origin"]

        info = _get_memory_usage()
        info["round_number"] = round_num
        info["origin"] = origin

        logger.info(f"Round {round_num}: Memory usage: {info}")
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"{origin}_memory_usage.json"), "a") as f:
                f.write(json.dumps(info) + "\n")


def _get_memory_usage() -> dict:
    process = psutil.Process(os.getpid())
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    info = {
        "process_memory": round(process.memory_info().rss / (1024**2), 2),
        "virtual_memory/total": round(virtual_memory.total / (1024**2), 2),
        "virtual_memory/available": round(virtual_memory.available / (1024**2), 2),
        "virtual_memory/percent": virtual_memory.percent,
        "virtual_memory/used": round(virtual_memory.used / (1024**2), 2),
        "virtual_memory/free": round(virtual_memory.free / (1024**2), 2),
        "virtual_memory/active": round(virtual_memory.active / (1024**2), 2),
        "virtual_memory/inactive": round(virtual_memory.inactive / (1024**2), 2),
        "virtual_memory/buffers": round(virtual_memory.buffers / (1024**2), 2),
        "virtual_memory/cached": round(virtual_memory.cached / (1024**2), 2),
        "virtual_memory/shared": round(virtual_memory.shared / (1024**2), 2),
        "swap_memory/total": round(swap_memory.total / (1024**2), 2),
        "swap_memory/used": round(swap_memory.used / (1024**2), 2),
        "swap_memory/free": round(swap_memory.free / (1024**2), 2),
        "swap_memory/percent": swap_memory.percent,
    }
    return info
