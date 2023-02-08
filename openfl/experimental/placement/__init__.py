# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.placement package."""

from .placement import RayExecutor, make_remote, aggregator, collaborator

__all__ = ["RayExecutor", "make_remote", "aggregator", "collaborator"]
