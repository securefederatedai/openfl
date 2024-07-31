# Copyright 2020-2024 Intel Corporation
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
    """
    Custom Thread class.

    This class extends the `threading.Thread` class and allows for the storage
    of the result returned by the target function.

    Attributes:
        target (function): The function to be executed in a separate thread.
        name (str): The name of the thread.
        args (tuple): The positional arguments to pass to the target function.
        kwargs (dict): The keyword arguments to pass to the target function.
    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        """Initialize a CustomThread object.

        Args:
            group (None, optional): Reserved for future extension when a
                ThreadGroup class is implemented.
            target (function, optional): The function to be executed in a
                separate thread.
            name (str, optional): The name of the thread.
            args (tuple, optional): The positional arguments to pass to the
                target function.
            kwargs (dict, optional): The keyword arguments to pass to the
                target function.
        """
        Thread.__init__(self, group, target, name, args, kwargs)
        self._result = None

    def run(self):
        """
        Method representing the thread's activity.

        This method is invoked by `thread.start()`.
        """
        if self._target is not None:
            self._result = self._target(*self._args, **self._kwargs)

    def result(self):
        """Get the result of the thread's activity.

        Returns:
            Any: The result of the target function.
        """
        return self._result


class PrepareTask:
    """
    `PrepareTask` class stores the decorated function metadata and instantiates
    either the `asyncio` or `thread` tasks to handle asynchronous
    and synchronous execution of the decorated function respectively.

    Attributes:
    target (function): decorated function
    timeout (int): Timeout duration in second(s).
    *args (tuple): Arguments passed as a parameter to decorated function.
    **kwargs (dict): Keyword arguments passed as a parameter to decorated function.
    """

    def __init__(self, target_fn, timeout, args, kwargs) -> None:
        """Initialize a PrepareTask object.

        Args:
            target_fn (function): The function to be executed.
            timeout (int): The maximum time to allow for the function's
                execution.
            args (tuple): The positional arguments to pass to the function.
            kwargs (dict): The keyword arguments to pass to the function.
        """
        self._target_fn = target_fn
        self._fn_name = target_fn.__name__
        self._max_timeout = timeout
        self._args = args
        self._kwargs = kwargs

    async def async_execute(self):
        """
        Execute the task asynchronously of the decorated function referenced
        by `self._target_fn`.

        Raises:
            asyncio.TimeoutError: If the execution exceeds the maximum time.
            Exception: If any other error occurs during execution.

        Returns:
            Any: The result of the function's execution.
                The returned value from `task.results()` depends on the
                decorated function.
        """
        task = asyncio.create_task(
            name=self._fn_name,
            coro=self._target_fn(*self._args, **self._kwargs),
        )

        try:
            await asyncio.wait_for(task, timeout=self._max_timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Timeout after {self._max_timeout} second(s), "
                f"Exception method: ({self._fn_name})"
            )
        except Exception:
            raise Exception(f"Generic Exception: {self._fn_name}")

        return task.result()

    def sync_execute(self):
        """Handles synchronous execution of the
        decorated function referenced by `self._target_fn`.

        Raises:
            TimeoutError: If the execution exceeds the maximum time.

        Returns:
            Any: The returned value from `task.results()` depends on the decorated function.
        """
        task = CustomThread(
            target=self._target_fn,
            name=self._fn_name,
            args=self._args,
            kwargs=self._kwargs,
        )
        task.start()
        # Execution continues if the decorated function completes within the timelimit.
        # If the execution exceeds time limit then the spawned thread is force
        # joined to current/main thread.
        task.join(self._max_timeout)

        # If the control is back to current/main thread
        # and the spawned thread is still alive then timeout and raise
        # exception.
        if task.is_alive():
            raise TimeoutError(
                f"Timeout after {self._max_timeout} second(s), "
                f"Exception method: ({self._fn_name})"
            )

        return task.result()


class SyncAsyncTaskDecoFactory:
    """
    Sync and Async Task decorator factory.

    This class is a factory for creating decorators for synchronous and
    asynchronous tasks.
    Task decorator factory allows creation of concrete implementation of
    `wrapper` interface     and `contextmanager` to setup a common
    functionality/resources shared by `async_wrapper` and `sync_wrapper`.

    Attributes:
        is_coroutine (bool): Whether the decorated function is a coroutine.
    """

    @contextmanager
    def wrapper(self, func, *args, **kwargs):
        """Create a context for the decorated function.

        Args:
            func (function): The function to be decorated.
            args (tuple): The positional arguments to pass to the function.
            kwargs (dict): The keyword arguments to pass to the function.

        Yields:
            None
        """
        yield

    def __call__(self, func):
        """
        Decorate the function. Call to `@fedtiming()` executes `__call__()`
        method delegated from the derived class `fedtiming` implementing
        `SyncAsyncTaskDecoFactory`.

        Args:
            func (function): The function to be decorated.

        Returns:
            function: The decorated function.
        """
        # Closures
        self.is_coroutine = asyncio.iscoroutinefunction(func)
        str_fmt = "{} Method ({}); Co-routine {}"

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            """
            Wrapper for synchronous execution of decorated function.
            """
            logger.debug(str_fmt.format("sync", func.__name__, self.is_coroutine))
            with self.wrapper(func, *args, **kwargs):
                return self.task.sync_execute()

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            """
            Wrapper for asynchronous execution of decorated function.
            """
            logger.debug(str_fmt.format("async", func.__name__, self.is_coroutine))
            with self.wrapper(func, *args, **kwargs):
                return await self.task.async_execute()

        # Identify if the decorated function is `async` or `sync` and return
        # appropriate wrapper.
        if self.is_coroutine:
            return async_wrapper
        return sync_wrapper


class fedtiming(SyncAsyncTaskDecoFactory):  # noqa: N801
    """FedTiming decorator factory.

    This class is a factory for creating decorators for timing synchronous and
    asynchronous tasks.

    Attributes:
        timeout (int): The maximum time to allow for the function's execution.
    """

    def __init__(self, timeout):
        """Initialize a FedTiming object.

        Args:
            timeout (int): The maximum time to allow for the function's
            execution.
        """
        self.timeout = timeout

    @contextmanager
    def wrapper(self, func, *args, **kwargs):
        """
        Create a context for the decorated function.

        This method sets up the task for execution and measures its execution
        time.
        Yields the control back to `async_wrapper` or `sync_wrapper` function
        call.

        Args:
            func (function): The function to be decorated.
            args (tuple): The positional arguments to pass to the function.
            kwargs (dict): The keyword arguments to pass to the function.

        Yields:
            None

        Raises:
            Exception: If an error occurs during the function's execution
                raised by `async_wrapper` or `sync_wrapper` and terminates the
                execution..
        """
        self.task = PrepareTask(target_fn=func, timeout=self.timeout, args=args, kwargs=kwargs)
        try:
            start = time.perf_counter()
            yield
            logger.info(f"({self.task._fn_name}) Elapsed Time: {time.perf_counter() - start}")
        except Exception as e:
            logger.exception(
                f"An exception of type {type(e).__name__} occurred. " f"Arguments:\n{e.args[0]!r}"
            )
            os._exit(status=os.EX_TEMPFAIL)
