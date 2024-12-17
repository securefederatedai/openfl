# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.utilities.data package."""

from openfl.utilities.data_splitters.data_splitter import DataSplitter
from openfl.utilities.data_splitters.numpy import (
    DirichletNumPyDataSplitter,
    EqualNumPyDataSplitter,
    LogNormalNumPyDataSplitter,
    NumPyDataSplitter,
    RandomNumPyDataSplitter,
)

__all__ = [
    "DataSplitter",
    "DirichletNumPyDataSplitter",
    "EqualNumPyDataSplitter",
    "LogNormalNumPyDataSplitter",
    "NumPyDataSplitter",
    "RandomNumPyDataSplitter",
]
