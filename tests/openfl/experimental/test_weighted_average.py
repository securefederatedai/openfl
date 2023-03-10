# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test for openfl.experimenal.interface.torch.aggregation_function module."""

from openfl.experimental.interface.torch import WeightedAverage
from openfl.federated.task.runner_pt import _get_optimizer_state

import torch as pt
import numpy as np

import pickle
from typing import List, Dict, Any
from copy import deepcopy


class Net(pt.nn.Module):
    """
    Returns a simple model for test case
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.linear1 = pt.nn.Linear(10, 20)
        self.linear2 = pt.nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def get_optimizer() -> Any:
    """
    Get optimizer
    """
    with open("optimizer.pth", "rb") as f:
        return pickle.load(f)


def take_model_weighted_average(models_state_dicts_list: List[Dict],
                                weights_list: List[int]) -> Dict:
    """
    Take models weighted average manually
    """
    tmp_list = []
    for model_state_dict in models_state_dicts_list:
        tmp_list.append(np.array([value.detach() for value in model_state_dict.values()],
                                 dtype=object))

    new_params = np.average(tmp_list, weights=weights_list, axis=0)

    new_state = {}
    for i, k in enumerate(models_state_dicts_list[0].keys()):
        new_state[k] = pt.from_numpy(new_params[i].numpy())
    return new_state


def take_optimizer_weighted_average(optimizer_state_dicts_list: List[Dict],
                                    weights_list: List[int]) -> Dict:
    """
    Take models weighted average manually
    """
    for optimizer_state_dict in optimizer_state_dicts_list:
        assert optimizer_state_dict.pop("__opt_state_needed") == "true"

    tmp_list = []
    for optimizer_state_dict in optimizer_state_dicts_list:
        tmp_list.append(np.array(list(optimizer_state_dict.values()), dtype=object))

    new_params = np.average(tmp_list, weights=weights_list, axis=0)

    new_state = {}
    for i, k in enumerate(optimizer_state_dicts_list[0].keys()):
        new_state[k] = new_params[i]
    return new_state


def test_list_weighted_average():
    """
    Test list weighted average
    """
    float_element_list = [0.4, 0.21, 0.1, 0.03]
    weights_list = [0.1, 0.25, 0.325, 0.325]

    weighted_average = WeightedAverage()

    averaged_loss_using_class = weighted_average(deepcopy(float_element_list),
                                                 weights_list)
    averaged_loss_manually = np.average(deepcopy(float_element_list),
                                        weights=weights_list, axis=0)

    assert np.all(averaged_loss_using_class) == np.all(averaged_loss_manually)


def test_model_weighted_average():
    """
    Test model weighted average
    """
    model_state_dicts_list = [Net().state_dict() for _ in range(4)]
    weights_list = [0.1, 0.25, 0.325, 0.325]

    weighted_average = WeightedAverage()

    averaged_model_using_class = weighted_average(deepcopy(model_state_dicts_list),
                                                  weights_list)
    averaged_model_manually = take_model_weighted_average(deepcopy(model_state_dicts_list),
                                                          weights_list)

    assert all(averaged_model_using_class) == all(averaged_model_manually)


def test_optimizer_weighted_average():
    """
    Test optimizers weighted average
    """
    optimizer_state_dicts_list = [_get_optimizer_state(get_optimizer()) for _ in range(4)]
    weights_list = [0.1, 0.25, 0.325, 0.325]

    weighted_average = WeightedAverage()

    averaged_optimizer_using_class = weighted_average(deepcopy(optimizer_state_dicts_list),
                                                      weights_list)
    averaged_optimizer_manually = take_optimizer_weighted_average(
        deepcopy(optimizer_state_dicts_list), weights_list)

    assert all(averaged_optimizer_using_class) == all(averaged_optimizer_manually)
