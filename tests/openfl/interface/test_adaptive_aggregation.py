# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""AdaptiveAggregation tests."""

from unittest import mock

import pytest

from openfl.interface.aggregation_functions.core.adaptive_aggregation import AdaptiveAggregation


@pytest.fixture
def agg():
    optimizer = mock.Mock
    agg_func = mock.Mock

    return AdaptiveAggregation(optimizer, agg_func)


@pytest.fixture
def db_iterator():
    return [{
            'round': 1,
            'tensor_name': 'tensor1',
            'tags': ('aggregated'),
            'nparray': [1]
            }]


def test_call_exception(agg, db_iterator):
    agg.optimizer.params = {'tensor1': 1, 'tensor2': 2}
    with pytest.raises(Exception):
        agg.call(['local_tensors'], db_iterator, 'tensor2', 1, ('tag'))


def test_call_default_agg_func(agg, db_iterator):
    agg.optimizer.params = {'tensor1': 1}
    agg.default_agg_func = mock.Mock(return_value='default_agg_func')
    assert agg.call(['local_tensors'], db_iterator, 'tensor2', 1,
                    ('tag')) == 'default_agg_func'


def test_call_successful(agg, db_iterator):
    agg.optimizer.params = {'tensor1': 1, 'tensor2': 2}
    agg._make_gradient = mock.Mock()
    agg.optimizer.step = mock.Mock()
    assert agg.call(['local_tensors'], db_iterator, 'tensor1', 1,
                    ('tag')) == 1
