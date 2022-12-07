# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.placement package."""

from .placement import ray_call_put, make_remote, aggregator, collaborator

__all__ = ["ray_call_put", "make_remote", "aggregator", "collaborator"]
