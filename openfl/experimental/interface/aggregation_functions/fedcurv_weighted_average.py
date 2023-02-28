"""Supported aggregation functions."""

import collections
import numpy as np
import torch

from openfl.interface.aggregation_functions.weighted_average import weighted_average as wa

def fedcurv_weighted_average(tensors, weights):
    """
    Aggregation function of FedCurv algorithm.
    Applies weighted average aggregation to all tensors
    except Fisher matrices variables (u_t, v_t).
    These variables are summed without weights.
    FedCurv paper: https://arxiv.org/pdf/1910.07796.pdf

    Args:
        tensors: Models state_dict list or optimizers state_dict list or loss list or accuracy list
        weights: Weight for each element in the list

    Returns:
        dict: Incase model list / optimizer list OR
        float: Incase of loss list or accuracy list
    """
    # Check the type of first element of tensors list
    if type(tensors[0]) in (dict, collections.OrderedDict):
        tmp_state_dict = {}
        input_state_dict_keys = tensors[0].keys()
        
        # Use diag elements of Fisher matrix
        for key in input_state_dict_keys:
            if (key[-2:] == '_u' or key[-2:] == '_v' or key[-2:] == '_w'):
                tmp_state_dict[key] = np.sum([tensor[key].detach().cpu() if type(tensor[key]) is torch.Tensor else tensor[key].cpu() for tensor in tensors],axis=0)
                continue
            tmp_state_dict[key] = np.average([tensor[key].detach().cpu() if type(tensor[key]) is torch.Tensor else tensor[key].cpu() for tensor in tensors], weights=weights, axis=0)
        return tmp_state_dict
    else:
        return wa(tensors, weights)
