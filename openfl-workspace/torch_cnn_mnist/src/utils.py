# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own utilities."""

from torch.utils.tensorboard import SummaryWriter


def write_metric(node_name, task_name, metric_name, metric, round_number):
    """Write metric callback."""
    writer = SummaryWriter('./logs/cnn_mnist', flush_secs=5)
    writer.add_scalar(f'{node_name}/{task_name}/{metric_name}', metric, round_number)
