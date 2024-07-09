
"""Data package."""

import importlib
from warnings import catch_warnings
from warnings import simplefilter

with catch_warnings():
    simplefilter(action='ignore', category=FutureWarning)
    if importlib.util.find_spec('tensorflow'):
        # ignore deprecation warnings in command-line interface
        import tensorflow  # NOQA

from openfl.federated.data.loader import DataLoader  # NOQA

if importlib.util.find_spec('tensorflow'):
    from openfl.federated.data.loader_tf import TensorFlowDataLoader  # NOQA
    from openfl.federated.data.loader_keras import KerasDataLoader  # NOQA
    from openfl.federated.data.federated_data import FederatedDataSet  # NOQA

if importlib.util.find_spec('torch'):
    from openfl.federated.data.loader_pt import PyTorchDataLoader  # NOQA
    from openfl.federated.data.federated_data import FederatedDataSet  # NOQA
