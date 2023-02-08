# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime package Runtime class."""

from .runtime import Runtime
from .local_runtime import LocalRuntime


__all__ = ["LocalRuntime", "Runtime"]
