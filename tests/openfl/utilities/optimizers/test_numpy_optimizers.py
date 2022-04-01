# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Numpy optimizers tests module."""

import numpy as np
import pytest

from openfl.utilities.optimizers.numpy.adagrad_optimizer import NumPyAdagrad
from openfl.utilities.optimizers.numpy.adam_optimizer import NumPyAdam
from openfl.utilities.optimizers.numpy.yogi_optimizer import NumPyYogi
from.func_for_optimization import mc_cormick_func
from.func_for_optimization import rosenbrock_func

EPS = 5e-5
START_POINT = {'x': np.array([0.0]), 'y': np.array([0.0])}


@pytest.mark.parametrize(
    'func,params,optim,num_iter', [
        (rosenbrock_func,
         START_POINT,
         NumPyAdagrad(learning_rate=0.08),
         5000),
        (rosenbrock_func,
         START_POINT,
         NumPyAdam(learning_rate=0.01),
         1000),
        (rosenbrock_func,
         START_POINT,
         NumPyYogi(learning_rate=0.01),
         1000),
        (mc_cormick_func,
         START_POINT,
         NumPyAdagrad(learning_rate=0.03),
         5000),
        (mc_cormick_func,
         START_POINT,
         NumPyAdam(learning_rate=0.01),
         1000),
        (mc_cormick_func,
         START_POINT,
         NumPyYogi(learning_rate=0.01),
         1000),
    ])
def test_opt(func, params, optim, num_iter):
    """Test optimizer by performing gradient descent iterations."""
    for i in range(num_iter):
        if i % 125 == 0:
            print(f'Iter: {i}', '\t',
                  f'current point: {params}',
                  '\t', f'func value={func(params)}')
        grads = func.get_grads(params)
        params = optim.step(params, grads)

    diff = np.array([params[param_name]
                     - func.true_answer[param_name] for param_name in params])
    diff = (diff**2).sum()  # calculate L2 norm
    assert diff <= EPS, f'Found parameters are not optimal, L2 difference: {diff}'
