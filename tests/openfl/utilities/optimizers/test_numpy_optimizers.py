# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Numpy optimizers tests module."""

import numpy as np
import pytest

from openfl.utilities.optimizers.numpy.adagrad_optimizer import NumpyAdagrad
from openfl.utilities.optimizers.numpy.adam_optimizer import NumpyAdam
from openfl.utilities.optimizers.numpy.yogi_optimizer import NumpyYogi
from.func_for_optimization import mc_cormick_func
from.func_for_optimization import rosenbrock_func

EPS = 5e-5


@pytest.mark.parametrize(
    'func,optim,num_iter', [
        (rosenbrock_func,
         NumpyAdagrad(params={'x': np.array([0]), 'y': np.array([0])},
                 learning_rate=0.08),
         5000),
        (rosenbrock_func,
         NumpyAdam(params={'x': np.array([0]), 'y': np.array([0])},
              learning_rate=0.01),
         1000),
        (rosenbrock_func,
         NumpyYogi(params={'x': np.array([0]), 'y': np.array([0])},
              learning_rate=0.01),
         1000),
        (mc_cormick_func,
         NumpyAdagrad(params={'x': np.array([0]), 'y': np.array([0])},
                 learning_rate=0.03),
         5000),
        (mc_cormick_func,
         NumpyAdam(params={'x': np.array([0]), 'y': np.array([0])},
              learning_rate=0.01),
         1000),
        (mc_cormick_func,
         NumpyYogi(params={'x': np.array([0]), 'y': np.array([0])},
              learning_rate=0.01),
         1000),
    ])
def test_opt(func, optim, num_iter):
    """Test optimizer by performing gradient descent iterations."""
    for i in range(num_iter):
        if i % 125 == 0:
            print(f'Iter: {i}', '\t',
                  f'current point: {optim.params}',
                  '\t', f'func value={func(optim.params)}')
        grads = func.get_grads(optim.params)
        optim.step(grads)

    diff = np.array([optim.params[param_name]
                     - func.true_answer[param_name] for param_name in optim.params])
    diff = (diff**2).sum()  # calculate L2 norm
    assert diff <= EPS, f'Found parameters are not optimal, L2 difference: {diff}'
