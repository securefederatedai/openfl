import importlib

if importlib.util.find_spec('tensorflow'):
    from openfl.utilities.optimizers.keras.fedprox import FedProxOptimizer # NOQA
