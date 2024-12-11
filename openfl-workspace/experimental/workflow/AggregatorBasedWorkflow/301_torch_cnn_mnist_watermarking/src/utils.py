# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from torch.utils.tensorboard import SummaryWriter


writer = None


def get_writer():
    """Create global writer object."""
    global writer
    if not writer:
        writer = SummaryWriter('./logs/cnn_mnist', flush_secs=5)
    return writer


def write_metric(node_name, task_name, metric_name, metric, round_number):
    """Write metric callback."""
    writer = get_writer()
    writer.add_scalar(f'{node_name}/{task_name}/{metric_name}', metric, round_number)
