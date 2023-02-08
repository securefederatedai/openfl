# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Components Timeout Configuration Module"""

import asyncio
import logging
import os
import time

from contextlib import contextmanager
from functools import wraps
from threading import Thread

logger = logging.getLogger(__name__)


class CustomThread(Thread):
    '''
    The CustomThread object implements `threading.Thread` class.
    Allows extensibility and stores the returned result from threaded execution.

    Attributes:
    target (function): decorated function
    name (str): Name of the decorated function
    *args (tuple): Arguments passed as a parameter to decorated function.
    **kwargs (dict): Keyword arguments passed as a parameter to decorated function.

    '''
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._result = None

    def run(self):
        '''
        `run()` Invoked by `thread.start()`
        '''
        if self._target is not None:
            self._result = self._target(*self._args, **self._kwargs)

    def result(self):
        return self._result


class PrepareTask():
    '''
    `PrepareTask` class stores the decorated function metadata and instantiates
    either the `asyncio` or `thread` tasks to handle asynchronous
    and synchronous execution of the decorated function respectively.

    Attributes:
    target (function): decorated function
    timeout (int): Timeout duration in second(s).
    *args (tuple): Arguments passed as a parameter to decorated function.
    **kwargs (dict): Keyword arguments passed as a parameter to decorated function.
    '''
    def __init__(self, target_fn, timeout, args, kwargs) -> None:
        self._target_fn = target_fn
        self._fn_name = target_fn.__name__
        self._max_timeout = timeout
        self._args = args
        self._kwargs = kwargs

    async def async_execute(self):
        '''Handles asynchronous execution of the
        decorated function referenced by `self._target_fn`.

        Raises:
            asyncio.TimeoutError: If the async execution exceeds permitted time limit.
            Exception: Captures generic exceptions.

        Returns:
            Any: The returned value from `task.results()` depends on the decorated function.
        '''
        task = asyncio.create_task(
            name=self._fn_name,
            coro=self._target_fn(*self._args, **self._kwargs)
        )

        try:
            await asyncio.wait_for(task, timeout=self._max_timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Timeout after {self._max_timeout} second(s), "
                                       f"Exception method: ({self._fn_name})")
        except Exception:
            raise Exception(f"Generic Exception: {self._fn_name}")

        return task.result()

    def sync_execute(self):
        '''Handles synchronous execution of the
        decorated function referenced by `self._target_fn`.

        Raises:
            TimeoutError: If the synchronous execution exceeds permitted time limit.

        Returns:
            Any: The returned value from `task.results()` depends on the decorated function.
        '''
        task = CustomThread(target=self._target_fn,
                            name=self._fn_name,
                            args=self._args,
                            kwargs=self._kwargs)
        task.start()
        # Execution continues if the decorated function completes within the timelimit.
        # If the execution exceeds time limit then
        # the spawned thread is force joined to current/main thread.
        task.join(self._max_timeout)

        # If the control is back to current/main thread
        # and the spawned thread is still alive then timeout and raise exception.
        if task.is_alive():
            raise TimeoutError(f"Timeout after {self._max_timeout} second(s), "
                               f"Exception method: ({self._fn_name})")

        return task.result()


class SyncAsyncTaskDecoFactory:
    '''
    `Sync` and `Async` Task decorator factory allows creation of
    concrete implementation of `wrapper` interface and `contextmanager` to
    setup a common functionality/resources shared by `async_wrapper` and `sync_wrapper`.

    '''

    @contextmanager
    def wrapper(self, func, *args, **kwargs):
        yield

    def __call__(self, func):
        '''
        Call to `@fedtiming()` executes `__call__()` method
        delegated from the derived class `fedtiming` implementing `SyncAsyncTaskDecoFactory`.
        '''

        # Closures
        self.is_coroutine = asyncio.iscoroutinefunction(func)
        str_fmt = "{} Method ({}); Co-routine {}"

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            '''
            Wrapper for synchronous execution of decorated function.
            '''
            logger.debug(str_fmt.format("sync", func.__name__, self.is_coroutine))
            with self.wrapper(func, *args, **kwargs):
                return self.task.sync_execute()

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            '''
            Wrapper for asynchronous execution of decorated function.
            '''
            logger.debug(str_fmt.format("async", func.__name__, self.is_coroutine))
            with self.wrapper(func, *args, **kwargs):
                return await self.task.async_execute()

        # Identify if the decorated function is `async` or `sync` and return appropriate wrapper.
        if self.is_coroutine:
            return async_wrapper
        return sync_wrapper


class fedtiming(SyncAsyncTaskDecoFactory):  # noqa: N801
    def __init__(self, timeout):
        self.timeout = timeout

    @contextmanager
    def wrapper(self, func, *args, **kwargs):
        '''
        Concrete implementation of setup and teardown logic, yields the control back to
        `async_wrapper` or `sync_wrapper` function call.

        Raises:
            Exception: Captures the exception raised by `async_wrapper`
            or `sync_wrapper` and terminates the execution.
        '''
        self.task = PrepareTask(
            target_fn=func,
            timeout=self.timeout,
            args=args,
            kwargs=kwargs
        )
        try:
            start = time.perf_counter()
            yield
            logger.info(f"({self.task._fn_name}) Elapsed Time: {time.perf_counter() - start}")
        except Exception as e:
            logger.exception(f"An exception of type {type(e).__name__} occurred. "
                             f"Arguments:\n{e.args[0]!r}")
            os._exit(status=os.EX_TEMPFAIL)
