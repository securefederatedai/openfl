import importlib

if importlib.util.find_spec('tensorflow') is not None:
    from openfl.utilities.optimizers.keras.fedprox import (
        FedProxOptimizer,  # NOQA
    )
