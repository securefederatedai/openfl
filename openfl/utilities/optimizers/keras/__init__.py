"""Keras optimizers package."""
import pkgutil
if pkgutil.find_loader('keras'):
    from .fedprox import FedProxOptimizer # NOQA
