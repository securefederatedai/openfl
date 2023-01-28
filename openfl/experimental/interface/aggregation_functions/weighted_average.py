"""Supported aggregation functions."""

import numpy as np


def weighted_average(models, weights):
    """Weighted average aggregation function."""
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        state_dict[key] = np.average([state[key] for state in state_dicts],
                                     weights=weights, axis=0)
    new_model.load_state_dict(state_dict)
    return new_model
