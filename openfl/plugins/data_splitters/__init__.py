"""openfl.plugins.data package."""
from openfl.plugins.data_splitters.data_splitter import DataSplitter
from openfl.plugins.data_splitters.numpy import DirichletNumPyDataSplitter
from openfl.plugins.data_splitters.numpy import EqualNumPyDataSplitter
from openfl.plugins.data_splitters.numpy import LogNormalNumPyDataSplitter
from openfl.plugins.data_splitters.numpy import NumPyDataSplitter
from openfl.plugins.data_splitters.numpy import RandomNumPyDataSplitter

__all__ = [
    'DataSplitter',
    'DirichletNumPyDataSplitter',
    'EqualNumPyDataSplitter',
    'LogNormalNumPyDataSplitter',
    'NumPyDataSplitter',
    'RandomNumPyDataSplitter',
]
