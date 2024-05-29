# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""openfl.experimental.utilities package."""

import inspect
import itertools
from types import MethodType

import numpy as np

from openfl.experimental.utilities import ResourcesAllocationError


def parse_attrs(ctx, exclude=[], reserved_words=["next", "runtime", "input"]):
    """Returns ctx attributes and artifacts"""
    # TODO Persist attributes to local disk, database, object store, etc. here
    cls_attrs = []
    valid_artifacts = []
    for i in inspect.getmembers(ctx):
        if (not hasattr(i[1], "task") and not i[0].startswith("_") and
                i[0] not in reserved_words and i[0] not in exclude and
                i not in inspect.getmembers(type(ctx))):
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
        assert isinstance(kwargs["include"], list)
        for in_attr in kwargs["include"]:
            if in_attr not in cls_attrs:
                raise RuntimeError(
                    f"argument '{in_attr}' not found in flow task {f.__name__}")
        for attr in cls_attrs:
            if attr not in kwargs["include"]:
                delattr(ctx, attr)
    elif "exclude" in kwargs:
        assert isinstance(kwargs["exclude"], list)
        for in_attr in kwargs["exclude"]:
            if in_attr not in cls_attrs:
                raise RuntimeError(
                    f"argument '{in_attr}' not found in flow task {f.__name__}")
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
        artifacts_iter, _ = generate_artifacts(
            ctx=ctx, reserved_words=chkpnt_reserved_words)
        task_id = ctx._metaflow_interface.create_task(parent_func.__name__)
        ctx._metaflow_interface.save_artifacts(
            artifacts_iter(),
            task_name=parent_func.__name__,
            task_id=task_id,
            buffer_out=step_stdout,
            buffer_err=step_stderr,
        )
        print(f"Saved data artifacts for {parent_func.__name__}")


def old_check_resource_allocation(num_gpus, each_participant_gpu_usage):
    remaining_gpu_memory = {}
    # TODO for each GPU the funtion tries see if all participant usages fit into a GPU, it it
    # doesn't it removes that
    # participant from the participant list, and adds it to the remaining_gpu_memory dict. So any
    # sum of GPU requirements above 1
    # triggers this.
    # But at this point the funtion will raise an error because remaining_gpu_memory is never
    # cleared.
    # The participant list should remove the participant if it fits in the gpu and save the
    # partipant if it doesn't and continue
    # to the next GPU to see if it fits in that one, only if we run out of GPUs should this
    # funtion raise an error.
    for gpu in np.ones(num_gpus, dtype=int):
        for i, (participant_name, participant_gpu_usage) in enumerate(
                each_participant_gpu_usage.items()):
            if gpu == 0:
                break
            if gpu < participant_gpu_usage:
                remaining_gpu_memory.update({participant_name: gpu})
                each_participant_gpu_usage = dict(
                    itertools.islice(each_participant_gpu_usage.items(), i))
            else:
                gpu -= participant_gpu_usage
    if len(remaining_gpu_memory) > 0:
        raise ResourcesAllocationError(
            f"Failed to allocate Participant {list(remaining_gpu_memory.keys())} "
            +
            "to specified GPU. Please try allocating lesser GPU resources to participants"
        )


def check_resource_allocation(num_gpus, each_participant_gpu_usage):
    # copy participant dict
    need_assigned = each_participant_gpu_usage.copy()
    # cycle through all available GPU availability
    for gpu in np.ones(num_gpus, dtype=int):
        # buffer to cycle though since need_assigned will change sizes as we assign participants
        current_dict = need_assigned.copy()
        for i, (participant_name,
                participant_gpu_usage) in enumerate(current_dict.items()):
            if gpu == 0:
                break
            if gpu < participant_gpu_usage:
                # participant doesn't fitm break to next GPU
                break
            else:
                # if participant fits remove from need_assigned
                need_assigned.pop(participant_name)
                gpu -= participant_gpu_usage

    # raise error if after going though all gpus there are still participants that needed to be
    # assigned
    if len(need_assigned) > 0:
        raise ResourcesAllocationError(
            f"Failed to allocate Participant {list(need_assigned.keys())} " +
            "to specified GPU. Please try allocating lesser GPU resources to participants"
        )
