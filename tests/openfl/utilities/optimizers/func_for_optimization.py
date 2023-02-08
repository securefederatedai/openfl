# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Numpy optimizers test functions module."""
from typing import Dict

import numpy as np


def rosenbrock_func(point: Dict[str, np.ndarray]) -> float:
    """
    Calculate Rosenbrock function.

    More details: https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    return (1 - point['x'])**2 + 100 * (point['y'] - point['x']**2)**2


def _get_rosenbrock_grads(point: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Calculate gradients for Rosenbrock function."""
    grads = {'x': np.array([0]), 'y': np.array([0])}
    grads['x'] = -2 * (1 - point['x']) - 400 * point['x'] * (point['y'] - point['x']**2)
    grads['y'] = grads['y'] + 200 * (point['y'] - point['x']**2)
    return grads


def mc_cormick_func(point: Dict[str, np.ndarray]) -> float:
    """
    Calculate McCormick function.

    More details: https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    return (np.sin(point['x'] + point['y'])
            + (point['x'] - point['y'])**2
            - 1.5 * point['x'] + 2.5 * point['y'] + 1)


def _get_mc_cormick_grads(point: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Calculate gradients for McCormick function."""
    grads = {'x': np.array([0]), 'y': np.array([0])}
    grads['x'] = np.cos(point['x'] + point['y']) + 2 * (point['x'] - point['y']) - 1.5
    grads['y'] = np.cos(point['x'] + point['y']) - 2 * (point['x'] - point['y']) + 2.5
    return grads


rosenbrock_func.get_grads = _get_rosenbrock_grads
rosenbrock_func.true_answer = {'x': np.array([1.0]), 'y': np.array([1.0])}

mc_cormick_func.get_grads = _get_mc_cormick_grads
mc_cormick_func.true_answer = {'x': np.array([-0.54719]), 'y': np.array([-1.54719])}

__all__ = [
    'rosenbrock_func',
    'mc_cormick_func',
]
