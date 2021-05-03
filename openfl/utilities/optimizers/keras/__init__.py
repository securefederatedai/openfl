"""Keras optimizers package."""
import pkgutil

if pkgutil.find_loader('tensorflow'):
    from .fedprox import FedProxOptimizer # NOQA
