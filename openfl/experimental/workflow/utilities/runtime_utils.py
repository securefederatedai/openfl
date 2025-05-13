# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.workflow.utilities package."""

import inspect
import itertools
from types import MethodType

import numpy as np

from openfl.experimental.workflow.utilities import ResourcesAllocationError


def parse_attrs(ctx, exclude=[], reserved_words=["next", "runtime", "input"]):
    """Parses the context to get its attributes and artifacts, excluding those
    specified.

    Args:
        ctx (any): The context to parse.
        exclude (list, optional): A list of attribute names to exclude.
            Defaults to an empty list.
        reserved_words (list, optional): A list of reserved words to exclude.
            Defaults to ["next", "runtime", "input"].

    Returns:
        tuple: A tuple containing a list of attribute names and a list of
            valid artifacts (pairs of attribute names and values).
    """
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


def generate_artifacts(
    ctx, reserved_words=["next", "runtime", "input", "checkpoint", "execute_task_args"]
):
    """Generates artifacts from the given context, excluding specified reserved
    words.

    Args:
        ctx (any): The context to generate artifacts from.
        reserved_words (list, optional): A list of reserved words to exclude.
            Defaults to ["next", "runtime", "input", "checkpoint", "execute_task_args"].

    Returns:
        tuple: A tuple containing a generator of artifacts and a list of
            attribute names.
    """
    cls_attrs, valid_artifacts = parse_attrs(ctx, reserved_words=reserved_words)

    def artifacts_iter():
        # Helper function from metaflow source
        while valid_artifacts:
            var, val = valid_artifacts.pop()
            yield var, val

    return artifacts_iter, cls_attrs


def filter_attributes(ctx, f, **kwargs):
    """Filters out attributes from the next task in the flow based on inclusion
    or exclusion.

    Args:
        ctx (any): The context to filter attributes from.
        f (function): The next task function in the flow.
        **kwargs: Optional arguments that specify the 'include' or 'exclude'
            lists.

    Raises:
        RuntimeError: If both 'include' and 'exclude' are present, or if an
            attribute in 'include' or 'exclude' is not found in the context's
            attributes.
    """
    _, cls_attrs = generate_artifacts(ctx=ctx)

    # Validate input arguments
    _validate_include_exclude(kwargs, cls_attrs)

    # Process the 'include' or 'exclude' lists
    if "include" in kwargs:
        _process_inclusion(ctx, cls_attrs, kwargs["include"], f)
    elif "exclude" in kwargs:
        _process_exclusion(ctx, cls_attrs, kwargs["exclude"], f)


def _validate_include_exclude(kwargs, cls_attrs):
    """Validates that 'include' and 'exclude' are not both present, and that
    attributes in 'include' or 'exclude' exist in the context.

    Args:
        kwargs (dict): The keyword arguments passed to filter_attributes.
        cls_attrs (list): The list of all attributes in the context.

    Raises:
        RuntimeError: If both 'include' and 'exclude' are present, or if any
            attribute in 'include' or 'exclude' does not exist in the context.
    """
    if "include" in kwargs and "exclude" in kwargs:
        raise RuntimeError("'include' and 'exclude' should not both be present")

    if "include" in kwargs:
        assert isinstance(kwargs["include"], list)
        for attr in kwargs["include"]:
            if attr not in cls_attrs:
                raise RuntimeError(f"Argument '{attr}' not found in flow task")

    if "exclude" in kwargs:
        assert isinstance(kwargs["exclude"], list)
        for attr in kwargs["exclude"]:
            if attr not in cls_attrs:
                raise RuntimeError(f"Argument '{attr}' not found in flow task")


def _process_inclusion(ctx, cls_attrs, include_list, f):
    """Handles the inclusion logic: removes attributes not in the 'include' list.

    Args:
        ctx (any): The context to filter attributes from.
        cls_attrs (list): The list of all attributes in the class.
        include_list (list): The list of attributes to keep.
        f (function): The task function to be processed.
    """
    for attr in cls_attrs:
        if attr not in include_list:
            delattr(ctx, attr)


def _process_exclusion(ctx, cls_attrs, exclude_list, f):
    """Handles the exclusion logic: removes attributes in the 'exclude' list.

    Args:
        ctx (any): The context to filter attributes from.
        cls_attrs (list): The list of all attributes in the class.
        exclude_list (list): The list of attributes to exclude.
        f (function): The task function to be processed.
    """
    for attr in cls_attrs:
        if attr in exclude_list and hasattr(ctx, attr):
            delattr(ctx, attr)


def checkpoint(ctx, parent_func, chkpnt_reserved_words=["next", "runtime", "execute_task_args"]):
    """Optionally saves the current state for the task just executed.

    Args:
        ctx (any): The context to checkpoint.
        parent_func (function): The function that was just executed.
        chkpnt_reserved_words (list, optional): A list of reserved words to
            exclude from checkpointing. Defaults to ["next", "runtime", "execute_task_args"].

    Returns:
        step_stdout (io.StringIO): parent_func stdout
    """

    # Extract the stdout & stderr from the buffer
    # NOTE: Any prints in this method before this line will be recorded as
    # step output/error
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
        return step_stdout


def old_check_resource_allocation(num_gpus, each_participant_gpu_usage):
    remaining_gpu_memory = {}
    # TODO for each GPU the function tries see if all participant usages fit
    # into a GPU, it it doesn't it removes that participant from the
    # participant list, and adds it to the remaining_gpu_memory dict. So any
    # sum of GPU requirements above 1 triggers this.
    # But at this point the function will raise an error because
    # remaining_gpu_memory is never cleared.
    # The participant list should remove the participant if it fits in the gpu
    # and save the partipant if it doesn't and continue to the next GPU to see
    # if it fits in that one, only if we run out of GPUs should this function
    # raise an error.
    for gpu in np.ones(num_gpus, dtype=int):
        for i, (participant_name, participant_gpu_usage) in enumerate(
            each_participant_gpu_usage.items()
        ):
            if gpu == 0:
                break
            if gpu < participant_gpu_usage:
                remaining_gpu_memory.update({participant_name: gpu})
                each_participant_gpu_usage = dict(
                    itertools.islice(each_participant_gpu_usage.items(), i)
                )
            else:
                gpu -= participant_gpu_usage
    if len(remaining_gpu_memory) > 0:
        raise ResourcesAllocationError(
            f"Failed to allocate Participant {list(remaining_gpu_memory.keys())} "
            + "to specified GPU. Please try allocating lesser GPU resources to participants"
        )


def check_resource_allocation(num_gpus, each_participant_gpu_usage):
    # copy participant dict
    need_assigned = each_participant_gpu_usage.copy()
    # cycle through all available GPU availability
    for gpu in np.ones(num_gpus, dtype=int):
        # buffer to cycle though since need_assigned will change sizes as we
        # assign participants
        current_dict = need_assigned.copy()
        for participant_name, participant_gpu_usage in current_dict.items():
            if gpu == 0:
                break
            if gpu < participant_gpu_usage:
                # participant doesn't fitm break to next GPU
                break
            else:
                # if participant fits remove from need_assigned
                need_assigned.pop(participant_name)
                gpu -= participant_gpu_usage

    # raise error if after going though all gpus there are still participants
    # that needed to be assigned
    if len(need_assigned) > 0:
        raise ResourcesAllocationError(
            f"Failed to allocate Participant {list(need_assigned.keys())} "
            + "to specified GPU. Please try allocating lesser GPU resources to participants"
        )
