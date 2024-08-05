# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Generic check functions."""


def check_type(obj, expected_type, logger):
    """Assert `obj` is of `expected_type` type.

    Args:
        obj (Any): The object to check.
        expected_type (type): The expected type of the object.
        logger (Logger): The logger to use for reporting the error.

    Raises:
        TypeError: If the object is not of the expected type.
    """
    if not isinstance(obj, expected_type):
        exception = TypeError(f"Expected type {type(obj)}, got type {str(expected_type)}")
        logger.exception(repr(exception))
        raise exception


def check_equal(x, y, logger):
    """Assert `x` and `y` are equal.

    Args:
        x (Any): The first value to compare.
        y (Any): The second value to compare.
        logger (Logger): The logger to use for reporting the error.

    Raises:
        ValueError: If the values are not equal.
    """
    if x != y:
        exception = ValueError(f"{x} != {y}")
        logger.exception(repr(exception))
        raise exception


def check_not_equal(x, y, logger, name="None provided"):
    """
    Assert `x` and `y` are not equal.

    Args:
        x (Any): The first value to compare.
        y (Any): The second value to compare.
        logger (Logger): The logger to use for reporting the error.
        name (str, optional): The name of the values. Defaults to
            'None provided'.

    Raises:
        ValueError: If the values are equal.
    """
    if x == y:
        exception = ValueError(f"Name {name}. Expected inequality, but {x} == {y}")
        logger.exception(repr(exception))
        raise exception


def check_is_in(element, _list, logger):
    """Assert `element` is in `_list` collection.

    Args:
        element (Any): The element to check.
        _list (Iterable): The collection to check in.
        logger (Logger): The logger to use for reporting the error.

    Raises:
        ValueError: If the element is not in the collection.
    """
    if element not in _list:
        exception = ValueError(f"Expected sequence membership, but {element} is not in {_list}")
        logger.exception(repr(exception))
        raise exception


def check_not_in(element, _list, logger):
    """Assert `element` is not in `_list` collection.

    Args:
        element (Any): The element to check.
        _list (Iterable): The collection to check in.
        logger (Logger): The logger to use for reporting the error.

    Raises:
        ValueError: If the element is in the collection.
    """
    if element in _list:
        exception = ValueError(f"Expected not in sequence, but {element} is in {_list}")
        logger.exception(repr(exception))
        raise exception
