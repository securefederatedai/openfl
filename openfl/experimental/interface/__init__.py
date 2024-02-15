# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.interface package."""

from .fl_spec import FLSpec
from .participants import Aggregator, Collaborator

__all__ = ["FLSpec", "Aggregator", "Collaborator"]
