# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Generic check functions."""


def check_type(obj, expected_type, logger):
    """Assert `obj` is of `expected_type` type."""
    if not isinstance(obj, expected_type):
        exception = TypeError(f'Expected type {type(obj)}, got type {str(expected_type)}')
        logger.exception(repr(exception))
        raise exception


def check_equal(x, y, logger):
    """Assert `x` and `y` are equal."""
    if x != y:
        exception = ValueError(f'{x} != {y}')
        logger.exception(repr(exception))
        raise exception


def check_not_equal(x, y, logger, name='None provided'):
    """Assert `x` and `y` are not equal."""
    if x == y:
        exception = ValueError(f'Name {name}. Expected inequality, but {x} == {y}')
        logger.exception(repr(exception))
        raise exception


def check_is_in(element, _list, logger):
    """Assert `element` is in `_list` collection."""
    if element not in _list:
        exception = ValueError(f'Expected sequence membership, but {element} is not in {_list}')
        logger.exception(repr(exception))
        raise exception


def check_not_in(element, _list, logger):
    """Assert `element` is not in `_list` collection."""
    if element in _list:
        exception = ValueError(f'Expected not in sequence, but {element} is in {_list}')
        logger.exception(repr(exception))
        raise exception
