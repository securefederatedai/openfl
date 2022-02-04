# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Numpy optimizers tests module."""
from typing import Dict

import numpy as np
import pytest

from openfl.utilities.optimizers.numpy.adagrad_optimizer import Adagrad

EPS = 5e-5


def rosenbrock_func(point: Dict[str, np.ndarray]) -> float:
    """
    Calculate Rosenbrock function.

    More details: https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    return (1 - point['x'])**2 + 100 * (point['y'] - point['x']**2)**2


def get_rosenbrock_grads(point: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Calculate gradients for Rosenbrock function."""
    grads = {'x': np.array([0]), 'y': np.array([0])}
    grads['x'] = -2 * (1 - point['x']) - 400 * point['x'] * (point['y'] - point['x']**2)
    grads['y'] = grads['y'] + 200 * (point['y'] - point['x']**2)
    return grads


rosenbrock_func.get_grads = get_rosenbrock_grads
rosenbrock_func.true_answer = {'x': np.array([1.0]), 'y': np.array([1.0])}


@pytest.mark.parametrize(
    'func,optim,num_iter', [
        (rosenbrock_func,
         Adagrad(params={'x': np.array([0]), 'y': np.array([0])},
                 learning_rate=0.08),
         5000),
    ])
def test_opt(func, optim, num_iter):
    """Test optimizer by performing gradient descent iterations."""
    for i in range(num_iter):
        if i % 250 == 0:
            print(f'Iter: {i}', '\t',
                  f'current point: {optim.params}',
                  '\t', f'func value={func(optim.params)}')
        grads = func.get_grads(optim.params)
        optim.step(grads)

    diff = np.array([optim.params[param_name]
                     - func.true_answer[param_name] for param_name in optim.params])
    diff = (diff**2).sum()  # calculate L2 norm
    assert diff <= EPS, f'Found parameters are not optimal, L2 difference: {diff}'
