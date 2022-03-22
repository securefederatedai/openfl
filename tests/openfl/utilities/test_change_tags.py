# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Data Splitters tests module."""
import pytest

from openfl.utilities import change_tags


@pytest.mark.parametrize(
    'tags,add_field,remove_field,expected_result', [
        ('abc', None, 'abc', ()),
        (('abc',), None, 'abc', ()),
        (['abc'], None, 'abc', ()),
        ('abc', 'def', None, ('abc', 'def')),
        (['abc'], 'ghi', None, ('abc', 'ghi')),
        (['abc', 'def'], 'ghi', None, ('abc', 'def', 'ghi')),
        (['abc', 'def'], None, 'abc', ('def',)),
        (['abc', 'def'], 'ghi', 'abc', ('def', 'ghi')),
        (('abc', 'def'), 'ghi', 'abc', ('def', 'ghi')),
        (('abc', 'def', 'def'), 'ghi', 'abc', ('def', 'ghi')),
        (('abc', 'ghi', 'def'), 'ghi', 'abc', ('ghi', 'def')),
    ])
def test_equal(tags, add_field, remove_field, expected_result):
    """Test equal splitter."""
    result = change_tags(tags, add_field, remove_field)
    assert result == expected_result
