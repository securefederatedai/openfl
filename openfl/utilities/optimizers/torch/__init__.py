"""PyTorch optimizers package."""
import pkgutil

if pkgutil.find_loader('torch'):
    from .fedprox import FedProxOptimizer # NOQA
    from .fedprox import AdamProx # NOQA
