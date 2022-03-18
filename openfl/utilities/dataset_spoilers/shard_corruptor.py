# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shard corruptor module."""

from typing import Callable
from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


def corrupt_shard(spoil_method: Callable[[ShardDataset], ShardDataset]) -> Callable:
    """
    Corrupting Shard Descriptor wrapper.

    Decorating a Shard Descriptor (SD) class with this function will result in the following:
    1. `corrupt` parameter is added to the SD init function.
        The Envoy manager may enable corruption by putting `corrupt: true` to SD params.
    2. `spoil_method` passed to the wrapper will be used to spoil Shard Datasets of this Envoy
        if corruption is enabled.
    """
    def decorator_func(ShardDescriptorClass: ShardDescriptor) -> ShardDescriptor:
        # This decorator is aware of the chosen `spoil_method`
        class WrapperClass(ShardDescriptorClass):
            """Extended SD class that is able to incorporate corruption."""
            def __init__(self, corrupt: bool = False, **kwargs):
                self.corrupt = corrupt
                super().__init__(**kwargs)

            def get_dataset(self, *args, **kwargs):
                original_shard_dataset = super().get_dataset(*args, **kwargs)
                if not self.corrupt:
                    return original_shard_dataset

                return spoil_method(original_shard_dataset)

        return WrapperClass

    return decorator_func
