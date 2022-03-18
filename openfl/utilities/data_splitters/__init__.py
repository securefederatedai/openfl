# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0-

"""openfl.utilities.data package."""
from openfl.utilities.data_splitters.data_splitter import DataSplitter
from openfl.utilities.data_splitters.numpy import DirichletNumPyDataSplitter
from openfl.utilities.data_splitters.numpy import EqualNumPyDataSplitter
from openfl.utilities.data_splitters.numpy import LogNormalNumPyDataSplitter
from openfl.utilities.data_splitters.numpy import NumPyDataSplitter
from openfl.utilities.data_splitters.numpy import RandomNumPyDataSplitter

__all__ = [
    'DataSplitter',
    'DirichletNumPyDataSplitter',
    'EqualNumPyDataSplitter',
    'LogNormalNumPyDataSplitter',
    'NumPyDataSplitter',
    'RandomNumPyDataSplitter',
]
