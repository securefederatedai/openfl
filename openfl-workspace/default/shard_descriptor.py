# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Shard Descriptor template.

It is recommended to perform tensor manipulations using numpy.
"""

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class LocalShardDescriptor(ShardDescriptor):
    """Shard descriptor subclass."""

    def __init__(self, data_path: str, sample_shape: tuple, target_shape: tuple) -> None:
        """
        Initialize local Shard Descriptor.

        Parameters are arbitrary, set up the ShardDescriptor-related part
        of the envoy_config.yaml as you need.
        """
        super().__init__()
