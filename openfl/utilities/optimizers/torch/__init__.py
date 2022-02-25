"""PyTorch optimizers package."""
import pkgutil

if pkgutil.find_loader('torch'):
    from .fedprox import FedProxOptimizer # NOQA
    from .fedprox import FedProxAdam # NOQA
    from .optimdevice import optimizer_to # NOQA
