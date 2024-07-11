# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Exceptions that occur during service interaction."""


class ShardNotFoundError(Exception):
    """Indicates that director has no information about that shard."""
