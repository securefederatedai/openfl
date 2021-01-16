# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utilities module."""

import numpy as np


def split_tensor_dict_into_floats_and_non_floats(tensor_dict):
    """
    Split the tensor dictionary into float and non-floating point values.

    Splits a tensor dictionary into float and non-float values.

    Args:
        tensor_dict: A dictionary of tensors

    Returns:
        Two dictionaries: the first contains all of the floating point tensors
        and the second contains all of the non-floating point tensors

    """
    float_dict = {}
    non_float_dict = {}
    for k, v in tensor_dict.items():
        if np.issubdtype(v.dtype, np.floating):
            float_dict[k] = v
        else:
            non_float_dict[k] = v
    return float_dict, non_float_dict


def split_tensor_dict_into_supported_and_not_supported_types(
        tensor_dict, keep_types):
    """
    Split the tensor dictionary into supported and not supported types.

    Args:
        tensor_dict: A dictionary of tensors
        keep_types: An iterable of supported types
    Returns:
        Two dictionaries: the first contains all of the supported tensors
        and the second contains all of the not supported tensors

    """
    keep_dict = {}
    holdout_dict = {}
    for k, v in tensor_dict.items():
        if any([np.issubdtype(v.dtype, type_) for type_ in keep_types]):
            keep_dict[k] = v
        else:
            holdout_dict[k] = v
    return keep_dict, holdout_dict


def split_tensor_dict_for_holdouts(logger, tensor_dict,
                                   keep_types=(np.floating, np.integer),
                                   holdout_tensor_names=()):
    """
    Split a tensor according to tensor types.

    Args:
        logger: The log object
        tensor_dict: A dictionary of tensors
        keep_types: A list of types to keep in dictionary of tensors
        holdout_tensor_names: A list of tensor names to extract from the
         dictionary of tensors

    Returns:
        Two dictionaries: the first is the original tensor dictionary minus
        the holdout tenors and the second is a tensor dictionary with only the
        holdout tensors

    """
    # initialization
    tensors_to_send = tensor_dict.copy()
    holdout_tensors = {}

    # filter by-name tensors from tensors_to_send and add to holdout_tensors
    # (for ones not already held out becuase of their type)
    for tensor_name in holdout_tensor_names:
        if tensor_name not in holdout_tensors.keys():
            try:
                holdout_tensors[tensor_name] = tensors_to_send.pop(tensor_name)
            except KeyError:
                logger.warn('tried to remove tensor: {} not present in the'
                            ' tensor dict'.format(tensor_name))
                continue

    # filter holdout_types from tensors_to_send and add to holdout_tensors
    tensors_to_send, not_supported_tensors_dict = \
        split_tensor_dict_into_supported_and_not_supported_types(
            tensors_to_send, keep_types)
    holdout_tensors = {**holdout_tensors, **not_supported_tensors_dict}

    return tensors_to_send, holdout_tensors
