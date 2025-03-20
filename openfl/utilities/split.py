# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""split tensors module."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def split_tensor_dict_into_floats_and_non_floats(tensor_dict):
    """Split the tensor dictionary into float and non-floating point values.

    This function splits a tensor dictionary into two dictionaries: one
    containing all the floating point tensors and the other containing all the
    non-floating point tensors.

    Args:
        tensor_dict (dict): A dictionary of tensors.

    Returns:
        Tuple[dict, dict]: The first dictionary contains all of the floating
            point tensors and the second dictionary contains all of the
            non-floating point tensors.
    """
    float_dict = {}
    non_float_dict = {}
    for k, v in tensor_dict.items():
        if np.issubdtype(v.dtype, np.floating):
            float_dict[k] = v
        else:
            non_float_dict[k] = v
    return float_dict, non_float_dict


def split_tensor_dict_by_types(tensor_dict, keep_types):
    """Split the tensor dictionary into supported and not supported types.

    Args:
        tensor_dict (dict): A dictionary of tensors.
        keep_types (Iterable[type]): An iterable of supported types.

    Returns:
        Tuple[dict, dict]: The first dictionary contains all of the supported
            tensors and the second dictionary contains all of the not
            supported tensors.
    """
    keep_dict = {}
    holdout_dict = {}
    for k, v in tensor_dict.items():
        if any(np.issubdtype(v.dtype, type_) for type_ in keep_types):
            keep_dict[k] = v
        else:
            holdout_dict[k] = v
    return keep_dict, holdout_dict


def split_tensor_dict_for_holdouts(
    tensor_dict,
    keep_types=(np.floating, np.integer),
    holdout_tensor_names=(),
):
    """
    Split a tensor according to tensor types.

    This function splits a tensor dictionary into two dictionaries: one
    containing the tensors to send and the other containing the holdout
    tensors.

    Args:
        tensor_dict (dict): A dictionary of tensors.
        keep_types (Tuple[type, ...], optional): A tuple of types to keep in
            the dictionary of tensors. Defaults to (np.floating, np.integer).
        holdout_tensor_names (Iterable[str], optional): An iterable of tensor
            names to extract from the dictionary of tensors. Defaults to ().

    Returns:
        Tuple[dict, dict]: The first dictionary is the original tensor
            dictionary minus the holdout tensors and the second dictionary is
            a tensor dictionary with only the holdout tensors.
    """
    # initialization
    tensors_to_send = tensor_dict.copy()
    holdout_tensors = {}

    # filter by-name tensors from tensors_to_send and add to holdout_tensors
    # (for ones not already held out because of their type)
    for tensor_name in holdout_tensor_names:
        if tensor_name not in holdout_tensors.keys():
            try:
                holdout_tensors[tensor_name] = tensors_to_send.pop(tensor_name)
            except KeyError:
                logger.warning(
                    f"tried to remove tensor: {tensor_name} not present in the tensor dict"
                )
                continue

    # filter holdout_types from tensors_to_send and add to holdout_tensors
    tensors_to_send, not_supported_tensors_dict = split_tensor_dict_by_types(
        tensors_to_send, keep_types
    )
    holdout_tensors = {**holdout_tensors, **not_supported_tensors_dict}

    return tensors_to_send, holdout_tensors
