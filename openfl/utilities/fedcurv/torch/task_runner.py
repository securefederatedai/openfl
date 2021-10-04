"""FedCurv PyTorch Task Runner module."""
from openfl.federated.task.runner_pt import PyTorchTaskRunner
from openfl.utilities import Metric
from openfl.utilities.fedcurv.torch import FedCurv


class FedCurvPyTorchTaskRunner(PyTorchTaskRunner):
    """Task Runner version of FedCurv algorithm.

    In order to use FedCurv algorithm in CLI,
    you need to inherit from this class,
    call base constructor first in child constructor,
    and define model structure in `initialize_network` function.

    Also, aggregation function must be overriden by
    openfl.utilities.fedcurv.aggregation_function.FedCurvWeightedAverage
    """

    def __init__(self, importance, **kwargs):
        """Initialize.

        Args:
            importance: Lambda coefficient of FedCurv algorithm.
        """
        super().__init__(**kwargs)
        self.initialize_network()
        self.fedcurv = FedCurv(self, importance)

    def initialize_network(self):
        """Model definition function.

        Must be defined by user.
        """
        raise NotImplementedError

    def train_epoch(self, batch_generator):
        """Train single epoch."""
        metric_dict = self.fedcurv.train(self, batch_generator, self.optimizer, self.device, self.loss_fn)
        name, value = next(iter(metric_dict))
        return Metric(name, value)
