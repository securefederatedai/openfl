# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import time

from tensorboardX import SummaryWriter

from openfl.callbacks.callback import Callback

logger = logging.getLogger(__name__)


class MetricWriter(Callback):
    """Log scalar metrics at the end of each round.

    Attributes:
        log_dir: Path to write logs as lines of JSON. Defaults to `./logs`.
        use_tensorboard: If True, writes scalar summaries to TensorBoard under `log_dir`.
    """

    def __init__(self, log_dir: str = "./logs/", use_tensorboard: bool = True):
        super().__init__()
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard

        self._log_file_handle = None
        self._summary_writer = None
        self._round_start_time = None

    def on_experiment_begin(self, logs=None):
        """Open file handles for logging."""
        os.makedirs(self.log_dir, exist_ok=True)

        if not self._log_file_handle:
            self._log_file_handle = open(
                os.path.join(self.log_dir, self.params["origin"] + "_metrics.txt"), "a"
            )

        if self.use_tensorboard:
            self._summary_writer = SummaryWriter(
                os.path.join(self.log_dir, self.params["origin"] + "_tensorboard")
            )

    def on_round_end(self, round_num: int, logs=None):
        """Log metrics.

        Args:
            round_num: The current round number.
            logs: A key-value pair of scalar metrics.
        """
        logs = logs or {}
        elapsed_seconds = time.monotonic() - self._round_start_time
        metrics = {
            "round_number": round_num,
            "elapsed_seconds": elapsed_seconds,
            **logs,
        }
        logger.info(f"Round {round_num}: Metrics: {metrics}")

        self._log_file_handle.write(json.dumps(metrics) + "\n")
        self._log_file_handle.flush()

        if self._summary_writer:
            for key, value in metrics.items():
                self._summary_writer.add_scalar(key, value, round_num)
            self._summary_writer.flush()

    def on_experiment_end(self, logs=None):
        """Cleanup."""
        if self._log_file_handle:
            self._log_file_handle.close()
            self._log_file_handle = None

        if self._summary_writer:
            self._summary_writer.close()
            
    def on_round_begin(self, round_num: int, logs=None):
        self._round_start_time = time.monotonic()