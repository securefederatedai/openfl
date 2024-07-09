# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
