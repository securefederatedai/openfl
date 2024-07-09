"""PyTorch optimizers package."""
import importlib

if importlib.util.find_spec('torch'):
    from openfl.utilities.optimizers.torch.fedprox import FedProxOptimizer # NOQA
    from openfl.utilities.optimizers.torch.fedprox import FedProxAdam # NOQA
