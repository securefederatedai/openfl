# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Data Splitters tests module."""
import pytest

from openfl.utilities import change_tags


@pytest.mark.parametrize(
    'tags,expected_result', [
        ((), ()),
        (('abc',), ('abc',)),
        (('abc', 'def'), ('abc', 'def')),
    ])
def test_change_tags_without_add_or_remove(tags, expected_result):
    """Test change_tags without add or remove fields."""
    result = change_tags(tags)
    assert result == expected_result


@pytest.mark.parametrize(
    'tags,add_field,expected_result', [
        (('abc',), None, ('abc',)),
        (('abc',), '', ('', 'abc')),
        (('abc', 'def'), 'ghi', ('abc', 'def', 'ghi')),
        (('abc', 'def'), 'def', ('abc', 'def')),
    ])
def test_change_tags_add(tags, add_field, expected_result):
    """Test change_tags with add field."""
    result = change_tags(tags, add_field=add_field)
    assert result == expected_result


@pytest.mark.parametrize(
    'tags,remove_field,expected_result', [
        (('abc',), None, ('abc',)),
        (('abc', 'def'), 'def', ('abc',)),
    ])
def test_change_tags_remove(tags, remove_field, expected_result):
    """Test change_tags with remove field."""
    result = change_tags(tags, remove_field=remove_field)
    assert result == expected_result


@pytest.mark.parametrize(
    'tags,add_field,remove_field,expected_result', [
        (('abc', 'def'), None, None, ('abc', 'def')),
        (('abc', 'def'), 'ghi', 'abc', ('def', 'ghi')),
        (('abc', 'def'), None, 'abc', ('def',)),
        (('abc', 'def'), 'ghi', None, ('abc', 'def', 'ghi')),
        (('abc', 'def'), 'ghi', 'ghi', ('abc', 'def')),
        (('abc', 'ghi', 'def'), 'ghi', 'abc', ('def', 'ghi')),
        (('abc', 'ghi', 'def'), 'ghi', 'ghi', ('abc', 'def')),
        (('abc', 'def'), 'def', 'def', ('abc',)),
    ])
def test_change_tags_add_and_remove_both(tags, add_field, remove_field, expected_result):
    """Test change tags with both add and remove fields."""
    result = change_tags(tags, add_field=add_field, remove_field=remove_field)
    assert result == expected_result


@pytest.mark.parametrize(
    'tags,remove_field', [
        (('abc', 'def'), 'ghi'),
        (('abc',), ''),
    ])
def test_change_tags_remove_not_in_tags(tags, remove_field):
    """Test change_tags with remove field not in tags."""
    with pytest.raises(Exception):
        change_tags(tags, remove_field=remove_field)


@pytest.mark.parametrize(
    'tags,add_field,remove_field', [
        (('abc', 'def'), None, 'ghi'),
        (('abc', 'def'), 'xyz', 'ghi'),
    ])
def test_change_tags_add_remove_not_in_tags(tags, add_field, remove_field):
    """Test change_tags with add and remove field not in tags."""
    with pytest.raises(Exception):
        change_tags(tags, add_field=add_field, remove_field=remove_field)


@pytest.mark.parametrize(
    'tags,add_field,remove_field,expected_result', [
        (('abc', 'def', 'def'), 'def', None, ('abc', 'def')),
        (('abc', 'def', 'def'), None, 'def', ('abc',)),
        (('abc', 'def', 'def'), 'ghi', 'abc', ('def', 'ghi')),
    ])
def test_change_tags_duplicate_fields_in_tags(tags, add_field, remove_field, expected_result):
    """Test change tags with duplicate fields in tags."""
    result = change_tags(tags, add_field=add_field, remove_field=remove_field)
    assert result == expected_result
