# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Data Splitters tests module."""
import numpy as np
import pytest

from openfl.utilities.data_splitters import DirichletNumPyDataSplitter
from openfl.utilities.data_splitters import EqualNumPyDataSplitter
from openfl.utilities.data_splitters import LogNormalNumPyDataSplitter
from openfl.utilities.data_splitters import RandomNumPyDataSplitter

np.random.seed(0)
y_train = np.random.randint(0, 10, 1000)


@pytest.mark.parametrize(
    'num_collaborators,expected_result', [
        (10, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
        (11, [91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 90]),
    ])
def test_equal(num_collaborators, expected_result):
    """Test equal splitter."""
    splitter = EqualNumPyDataSplitter()
    shards = splitter.split(y_train, num_collaborators)
    assert [len(shard) for shard in shards] == expected_result


@pytest.mark.parametrize(
    'num_collaborators,expected_result', [
        (10, [18, 258, 110, 120, 26, 69, 70, 174, 94, 61]),
        (11, [18, 183, 75, 110, 120, 26, 69, 70, 174, 94, 61]),
    ])
def test_random(num_collaborators, expected_result):
    """Test random splitter."""
    splitter = RandomNumPyDataSplitter()
    shards = splitter.split(y_train, num_collaborators)
    print([len(shard) for shard in shards])
    assert [len(shard) for shard in shards] == expected_result


@pytest.mark.parametrize(
    'num_collaborators,expected_result', [
        (10, [154, 9, 33, 64, 85, 35, 48, 18, 26, 4]),
        (20, [36, 81, 35, 56, 105, 6, 114, 46, 57, 50, 24, 55, 30, 9, 14, 10, 15, 48, 12, 4]),
    ])
def test_lognormal(num_collaborators, expected_result):
    """Test lognormal splitter."""
    splitter = LogNormalNumPyDataSplitter(
        mu=1,
        sigma=1,
        num_classes=10,
        classes_per_col=2,
        min_samples_per_class=2
    )
    shards = splitter.split(y_train, num_collaborators)
    print([len(shard) for shard in shards])
    assert [len(shard) for shard in shards] == expected_result


@pytest.mark.parametrize(
    'num_collaborators,expected_result', [
        (10, [56, 51, 107, 64, 158, 122, 131, 103, 104, 104]),
        (11, [60, 95, 112, 111, 31, 106, 90, 96, 123, 97, 79]),
    ])
def test_dirichlet(num_collaborators, expected_result):
    """Test dirichlet splitter."""
    splitter = DirichletNumPyDataSplitter()
    shards = splitter.split(y_train, num_collaborators)
    print([len(shard) for shard in shards])
    assert [len(shard) for shard in shards] == expected_result
