
"""Task package."""

import importlib
from warnings import catch_warnings
from warnings import simplefilter

with catch_warnings():
    simplefilter(action='ignore', category=FutureWarning)
    if importlib.util.find_spec('tensorflow'):
        # ignore deprecation warnings in command-line interface
        import tensorflow  # NOQA

from openfl.federated.task.runner import TaskRunner  # NOQA


if importlib.util.find_spec('tensorflow'):
    from openfl.federated.task.runner_tf import TensorFlowTaskRunner  # NOQA
    from openfl.federated.task.runner_keras import KerasTaskRunner  # NOQA
    from openfl.federated.task.fl_model import FederatedModel  # NOQA
if importlib.util.find_spec('torch'):
    from openfl.federated.task.runner_pt import PyTorchTaskRunner  # NOQA
    from openfl.federated.task.fl_model import FederatedModel  # NOQA
