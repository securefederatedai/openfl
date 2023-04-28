# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.utilities package."""

import itertools
import inspect
import numpy as np
from types import MethodType
from openfl.experimental.utilities import ResourcesAllocationError


def parse_attrs(ctx, exclude=[], reserved_words=["next", "runtime", "input"]):
    """Returns ctx attributes and artifacts"""
    # TODO Persist attributes to local disk, database, object store, etc. here
    cls_attrs = []
    valid_artifacts = []
    for i in inspect.getmembers(ctx):
        if (
            not hasattr(i[1], "task")
            and not i[0].startswith("_")
            and i[0] not in reserved_words
            and i[0] not in exclude
            and i not in inspect.getmembers(type(ctx))
        ):
            if not isinstance(i[1], MethodType):
                cls_attrs.append(i[0])
                valid_artifacts.append((i[0], i[1]))
    return cls_attrs, valid_artifacts


def generate_artifacts(ctx, reserved_words=["next", "runtime", "input"]):
    """Returns ctx artifacts, and artifacts_iter method"""
    cls_attrs, valid_artifacts = parse_attrs(ctx, reserved_words=reserved_words)

    def artifacts_iter():
        # Helper function from metaflow source
        while valid_artifacts:
            var, val = valid_artifacts.pop()
            yield var, val

    return artifacts_iter, cls_attrs


def filter_attributes(ctx, f, **kwargs):
    """
    Filter out explicitly included / excluded attributes from the next task
    in the flow.
    """

    _, cls_attrs = generate_artifacts(ctx=ctx)
    if "include" in kwargs and "exclude" in kwargs:
        raise RuntimeError("'include' and 'exclude' should not both be present")
    elif "include" in kwargs:
        assert type(kwargs["include"]) == list
        for in_attr in kwargs["include"]:
            if in_attr not in cls_attrs:
                raise RuntimeError(
                    f"argument '{in_attr}' not found in flow task {f.__name__}"
                )
        for attr in cls_attrs:
            if attr not in kwargs["include"]:
                delattr(ctx, attr)
    elif "exclude" in kwargs:
        assert type(kwargs["exclude"]) == list
        for in_attr in kwargs["exclude"]:
            if in_attr not in cls_attrs:
                raise RuntimeError(
                    f"argument '{in_attr}' not found in flow task {f.__name__}"
                )
        for attr in cls_attrs:
            if attr in kwargs["exclude"] and hasattr(ctx, attr):
                delattr(ctx, attr)


def checkpoint(ctx, parent_func, chkpnt_reserved_words=["next", "runtime"]):
    """
    [Optionally] save current state for the task just executed task
    """

    # Extract the stdout & stderr from the buffer
    # NOTE: Any prints in this method before this line will be recorded as step output/error
    step_stdout, step_stderr = parent_func._stream_buffer.get_stdstream()

    if ctx._checkpoint:
        # all objects will be serialized using Metaflow interface
        print(f"Saving data artifacts for {parent_func.__name__}")
        artifacts_iter, _ = generate_artifacts(ctx=ctx, reserved_words=chkpnt_reserved_words)
        task_id = ctx._metaflow_interface.create_task(parent_func.__name__)
        ctx._metaflow_interface.save_artifacts(
            artifacts_iter(),
            task_name=parent_func.__name__,
            task_id=task_id,
            buffer_out=step_stdout,
            buffer_err=step_stderr,
        )
        print(f"Saved data artifacts for {parent_func.__name__}")

def check_resource_allocation(num_gpus, each_collab_gpu_usage):
    remaining_gpu_memory = {}
    for gpu in np.ones(num_gpus, dtype=int):
        for i, (collab_name, collab_gpu_usage) in enumerate(each_collab_gpu_usage.items()):
            if gpu == 0:
                break
            if gpu < collab_gpu_usage: # and collab_gpu_usage > 0:
                remaining_gpu_memory.update({collab_name: gpu})
                print (f"each_collab_gpu_usage: {each_collab_gpu_usage}")
                print (f"i: {i}")
                each_collab_gpu_usage = dict(itertools.islice(each_collab_gpu_usage.items(), i))
            else:
                gpu -= collab_gpu_usage
    print (f"len(remaining_gpu_memory): {len(remaining_gpu_memory)}")
    print (f"remaining_gpu_memory: {remaining_gpu_memory}")
    print (f"len(each_collab_gpu_usage): {len(each_collab_gpu_usage)}")
    if len(remaining_gpu_memory) > 0: # or len(each_collab_gpu_usage) > 1:
        raise ResourcesAllocationError(
            f"Failed to allocate Collaborator {list(remaining_gpu_memory.keys())} "
            + "to specified GPU. Please try allocating lesser GPU resources to collaborators"
        )
