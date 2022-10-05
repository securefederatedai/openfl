# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.placement package."""

from .placement import make_remote, aggregator, collaborator

__all__ = [
    'make_remote',
    'aggregator',
    'collaborator'
]
