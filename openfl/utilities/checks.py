# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Generic check functions."""


def check_type(obj, expected_type, logger):
    """Assert `obj` is of `expected_type` type."""
    if not isinstance(obj, expected_type):
        exception = TypeError(
            "Expected type {}, got type {}".format(
                type(obj), str(expected_type)
            )
        )
        logger.exception(repr(exception))
        raise exception


def check_equal(x, y, logger):
    """Assert `x` and `y` are equal."""
    if x != y:
        exception = ValueError("{} != {}".format(x, y))
        logger.exception(repr(exception))
        raise exception


def check_not_equal(x, y, logger, name='None provided'):
    """Assert `x` and `y` are not equal."""
    if x == y:
        exception = ValueError(
            "Name {}. Expected inequality, but {} == {}".format(name, x, y))
        logger.exception(repr(exception))
        raise exception


def check_is_in(element, _list, logger):
    """Assert `element` is in `_list` collection."""
    if element not in _list:
        exception = ValueError(
            "Expected sequence memebership, but {} is not in {}".format(
                element, _list))
        logger.exception(repr(exception))
        raise exception


def check_not_in(element, _list, logger):
    """Assert `element` is not in `_list` collection."""
    if element in _list:
        exception = ValueError(
            "Expected not in sequence, but {} is in {}".format(element, _list))
        logger.exception(repr(exception))
        raise exception
