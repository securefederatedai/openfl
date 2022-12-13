# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import ray
from copy import deepcopy
from openfl.experimental.utilities import (
    RedirectStdStreamContext,
    GPUResourcesNotAvailable,
    get_number_of_gpus,
)


class RayExecutor:
    def __init__(self):
        self.remote_functions = []
        self.remote_contexts = []

    def ray_call_put(self, ctx, func):
        remote_to_exec = make_remote(func, num_gpus=func.num_gpus)
        ref_ctx = ray.put(ctx)
        self.remote_contexts.append(ref_ctx)
        self.remote_functions.append(
            remote_to_exec.remote(ref_ctx, func.__name__)
        )
        del remote_to_exec
        del ref_ctx

    def get_remote_clones(self):
        clones = deepcopy(ray.get(self.remote_functions))
        del self.remote_functions
        # Remove clones from ray object store
        for ctx in self.remote_contexts:
            ray.cancel(ctx)
        return clones


def make_remote(f, num_gpus):
    f = ray.put(f)

    @functools.wraps(f)
    @ray.remote(num_gpus=num_gpus, max_calls=1)
    def wrapper(*args, **kwargs):
        f = getattr(args[0], args[1])
        print(f"\nRunning {f.__name__} in a new process")
        f()
        return args[0]

    return wrapper


def aggregator(f=None, *, num_gpus=0):
    if f is None:
        return functools.partial(aggregator, num_gpus=num_gpus)

    print(f'Aggregator step "{f.__name__}" registered')
    f.is_step = True
    f.decorators = []
    f.name = f.__name__
    f.task = True
    f.aggregator_step = True
    f.collaborator_step = False
    if f.__doc__:
        f.__doc__ = "<Node: Aggregator>" + f.__doc__
    total_gpus = get_number_of_gpus()
    if total_gpus < num_gpus:
        GPUResourcesNotAvailable(
            f"cannot assign more than available GPUs ({total_gpus} < {num_gpus})."
        )
    f.num_gpus = num_gpus

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print(f"\nCalling {f.__name__}")
        with RedirectStdStreamContext() as context_stream:
            # context_stream capture stdout and stderr for the function f.__name__
            setattr(wrapper, "_stream_buffer", context_stream)
            f(*args, **kwargs)

    return wrapper


def collaborator(f=None, *, num_gpus=0):
    if f is None:
        return functools.partial(collaborator, num_gpus=num_gpus)

    print(f'Collaborator step "{f.__name__}" registered')
    f.is_step = True
    f.decorators = []
    f.name = f.__name__
    f.task = True
    f.aggregator_step = False
    f.collaborator_step = True
    if f.__doc__:
        f.__doc__ = "<Node: Collaborator>" + f.__doc__
    total_gpus = get_number_of_gpus()
    if total_gpus < num_gpus:
        GPUResourcesNotAvailable(
            f"cannot assign more than available GPUs ({total_gpus} < {num_gpus})."
        )
    f.num_gpus = num_gpus

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print(f"\nCalling {f.__name__}")
        with RedirectStdStreamContext() as context_stream:
            # context_stream capture stdout and stderr for the function f.__name__
            setattr(wrapper, "_stream_buffer", context_stream)
            f(*args, **kwargs)

    return wrapper
