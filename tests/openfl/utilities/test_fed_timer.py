# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Federation Components Timeout tests module."""

import asyncio
import os
import pytest
import time

from openfl.utilities.fed_timer import fedtiming
from unittest import mock


@pytest.mark.parametrize(
    'input,expected', [
        ('first', 'first'),
    ])
def test_check_sync_function_return_same_value_within_timelimit(input, expected):
    """
    Test that the decorated synchronous function return the output within the timeout threshold
    Function call returns the expected output and it is asserted.
    """

    @fedtiming(timeout=0.2)
    def some_sync_fn(value):
        time.sleep(0.1)  # Simulate long running operation
        return value

    assert some_sync_fn(input) == expected


@pytest.mark.parametrize(
    'input,expected', [
        (True, True),
    ])
def test_check_async_function_return_same_value_within_timelimit(input, expected):
    """
    Test that the decorated asynchronous function return the output within the timeout threshold
    Function call returns the expected output and it is asserted
    """

    @fedtiming(timeout=0.2)
    async def some_async_fn(value):
        await asyncio.sleep(0.1)  # Simulate long running operation
        return value

    assert asyncio.run(some_async_fn(input)) == expected


def test_check_sync_function_timeout():
    """
    Test that the decorated synchronous function exceeds the timeout value and
    1. function call returns - None
    2. execution terminates and exit status is asserted
    """

    os._exit = mock.MagicMock()

    @fedtiming(timeout=0.1)
    def some_sync_fn(value):
        time.sleep(0.2)
        return value

    assert some_sync_fn('') is None
    assert os._exit.called


def test_check_async_function_timeout():
    """
    Test that the decorated asynchronous function exceeds the timeout value and
    1. function call returns - None
    2. execution terminates and exit status is asserted
    """

    os._exit = mock.MagicMock()

    @fedtiming(timeout=0.1)
    async def some_async_fn(value):
        await asyncio.sleep(0.2)
        return value

    assert asyncio.run(some_async_fn('')) is None
    assert os._exit.called


def test_check_sync_decorated_function_returns_normal_function():
    """
    Test if during compile time the decorated synchronous function is evaluated
    and returns a normal python function (sync_wrapper) from SyncAsyncTaskDecoFactory.
    """

    @fedtiming(timeout=0.1)
    def some_sync_fn():
        pass

    assert asyncio.iscoroutinefunction(some_sync_fn) is False
    assert some_sync_fn.__code__.co_name == 'sync_wrapper'


def test_check_async_decorated_function_returns_coroutine():
    """
    Test if during compile time the decorated async function is evaluated
    and returns a coroutine instance (async_wrapper) from SyncAsyncTaskDecoFactory.
    """

    @fedtiming(timeout=0.2)
    async def some_async_fn():
        await asyncio.sleep(0.1)

    assert asyncio.iscoroutinefunction(some_async_fn) is True
    assert some_async_fn.__code__.co_name == 'async_wrapper'
