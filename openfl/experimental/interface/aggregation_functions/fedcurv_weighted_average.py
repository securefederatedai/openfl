"""Supported aggregation functions."""

import numpy as np

def fedcurv_weighted_average(models, weights):
    """Aggregation function of FedCurv algorithm.
    Applies weighted average aggregation to all tensors
    except Fisher matrices variables (u_t, v_t).
    These variables are summed without weights.
    FedCurv paper: https://arxiv.org/pdf/1910.07796.pdf
    """
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        if (key[-2:] == '_u' or key[-2:] == '_v' or key[-2:] == '_w'):
            state_dict[key] = np.sum([state[key].cpu() for state in state_dicts],axis=0)
            continue
        state_dict[key] = np.average([state[key].cpu() for state in state_dicts], weights=weights, axis=0)
    new_model.load_state_dict(state_dict)
    return new_model
