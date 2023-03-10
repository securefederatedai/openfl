# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""openfl.experimental.interface.torch.aggregation_functions.weighted_average package."""

import collections
import numpy as np
import torch as pt


def weighted_average(tensors, weights):
    """Compute weighted average."""
    return np.average(tensors, weights=weights, axis=0)


class WeightedAverage:
    """Weighted average aggregation."""

    def __call__(self, objects_list, weights_list) -> np.ndarray:
        """
        Compute weighted average of models, optimizers, loss, or accuracy metrics.
        For taking weighted average of optimizer do the following steps:
        1.  Call "_get_optimizer_state" (openfl.federated.task.runner_pt._get_optimizer_state)
            pass optimizer to it, to take optimizer state dictionary.
        2.  Pass optimizer state dictionaries list to here.
        3.  To set the weighted average optimizer state dictionary back to optimizer,
            call "_set_optimizer_state" (openfl.federated.task.runner_pt._set_optimizer_state)
            and pass optimizer, device, and optimizer dictionary received in step 2.

        Args:
            objects_list: List of objects for which weighted average is to be computed.
            - List of Model state dictionaries , or
            - List of Metrics (Loss or accuracy), or
            - List of optimizer state dictionaries (following steps need to be performed)
                1. Obtain optimizer state dictionary by invoking "_get_optimizer_state"
                   (openfl.federated.task.runner_pt._get_optimizer_state).
                2. Create a list of optimizer state dictionary obtained in step - 1
                   Invoke WeightedAverage on this list.
                3. Invoke "_set_optimizer_state" to set weighted average of optimizer
                   state back to optimizer (openfl.federated.task.runner_pt._set_optimizer_state).
            weights_list: Weight for each element in the list.

        Returns:
            dict: For model or optimizer
            float: For Loss or Accuracy metrics
        """
        # Check the type of first element of tensors list
        if type(objects_list[0]) in (dict, collections.OrderedDict):
            optimizer = False
            # If __opt_state_needed found then optimizer state dictionary is passed
            if "__opt_state_needed" in objects_list[0]:
                optimizer = True
                # Remove __opt_state_needed from all state dictionary in list, and
                # check if weightedaverage of optimizer can be taken.
                for tensor in objects_list:
                    error_msg = "Optimizer is stateless, WeightedAverage cannot be taken"
                    assert tensor.pop("__opt_state_needed") == "true", error_msg

            tmp_list = []
            # # Take keys in order to rebuild the state dictionary taking keys back up
            for tensor in objects_list:
                # Append values of each state dictionary in list
                # If type(value) is Tensor then it needs to be detached
                tmp_list.append(np.array([value.detach() if isinstance(value, pt.Tensor) else value
                                          for value in tensor.values()], dtype=object))
            # Take weighted average of list of arrays
            # new_params passed is weighted average of each array in tmp_list
            new_params = weighted_average(tmp_list, weights_list)
            new_state = {}
            # Take weighted average parameters and building a dictionary
            for i, k in enumerate(objects_list[0].keys()):
                if optimizer:
                    new_state[k] = new_params[i]
                else:
                    new_state[k] = pt.from_numpy(new_params[i].numpy())
            return new_state
        else:
            return weighted_average(objects_list, weights_list)
