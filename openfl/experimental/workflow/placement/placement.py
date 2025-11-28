# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import functools
from typing import Callable

from openfl.experimental.workflow.utilities import RedirectStdStreamContext


def aggregator(f: Callable = None) -> Callable:
    """Placement decorator that designates that the task will run at the
    aggregator node.

    Usage:
    class MyFlow(FLSpec):
        ...
        @aggregator
        def agg_task(self):
           ...
        ...

    Args:
        f (Callable, optional): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """
    print(f'Aggregator step "{f.__name__}" registered')
    f.is_step = True
    f.decorators = []
    f.name = f.__name__
    f.task = True
    f.aggregator_step = True
    f.collaborator_step = False
    if f.__doc__:
        f.__doc__ = "<Node: Aggregator>" + f.__doc__

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print(f"\nCalling {f.__name__}")
        with RedirectStdStreamContext() as context_stream:
            # context_stream capture stdout and stderr for the function f.__name__
            wrapper._stream_buffer = context_stream
            f(*args, **kwargs)

    return wrapper


def collaborator(f: Callable = None) -> Callable:
    """Placement decorator that designates that the task will run at the
    collaborator node.

    Usage:
    class MyFlow(FLSpec):
        ...
        @collaborator
        def collaborator_task(self):
            ...

        @collaborator(num_gpus=1)
        @def collaborator_gpu_task:
            ...
        ...

    Args:
        f (Callable, optional): The function to be decorated.
        num_gpus (float, optional): [Applicable for Ray backend only]
                  Defines how many GPUs will be made available
                  to the task (Default = 0). Selecting a value < 1 (0.0-1.0]
                  will result in sharing of GPUs between tasks. 1 >= results in
                  exclusive GPU access for the task.

    Returns:
        Callable: The decorated function.
    """
    if f is None:
        return functools.partial(collaborator)

    print(f'Collaborator step "{f.__name__}" registered')
    f.is_step = True
    f.decorators = []
    f.name = f.__name__
    f.task = True
    f.aggregator_step = False
    f.collaborator_step = True
    if f.__doc__:
        f.__doc__ = "<Node: Collaborator>" + f.__doc__

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print(f"\nCalling {f.__name__}")
        with RedirectStdStreamContext() as context_stream:
            # context_stream capture stdout and stderr for the function f.__name__
            wrapper._stream_buffer = context_stream
            f(*args, **kwargs)

    return wrapper
