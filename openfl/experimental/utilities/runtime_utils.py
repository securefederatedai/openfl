# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.utilities package."""

import inspect
from types import MethodType


def parse_attrs(ctx, exclude=[], reserved_words=["next", "runtime", "input"]):
    """Parses the context to get its attributes and artifacts, excluding those specified.

    Args:
        ctx (any): The context to parse.
        exclude (list, optional): A list of attribute names to exclude. Defaults to an empty list.
        reserved_words (list, optional): A list of reserved words to exclude. Defaults to ["next", "runtime", "input"].

    Returns:
        tuple: A tuple containing a list of attribute names and a list of valid artifacts (pairs of attribute names and values).
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


def generate_artifacts(ctx, reserved_words=["next", "runtime", "input"]):
    """Generates artifacts from the given context, excluding specified reserved words.

    Args:
        ctx (any): The context to generate artifacts from.
        reserved_words (list, optional): A list of reserved words to exclude. Defaults to ["next", "runtime", "input"].

    Returns:
        tuple: A tuple containing a generator of artifacts and a list of attribute names.
    """
    cls_attrs, valid_artifacts = parse_attrs(ctx, reserved_words=reserved_words)

    def artifacts_iter():
        # Helper function from metaflow source
        while valid_artifacts:
            var, val = valid_artifacts.pop()
            yield var, val

    return artifacts_iter, cls_attrs


def filter_attributes(ctx, f, **kwargs):
    """
    Filters out attributes from the next task in the flow based on inclusion or exclusion.

    Args:
        ctx (any): The context to filter attributes from.
        f (function): The next task function in the flow.
        **kwargs: Optional arguments that specify the 'include' or 'exclude' lists.

    Raises:
        RuntimeError: If both 'include' and 'exclude' are present, or if an attribute in 'include' or 'exclude' is not 
            found in the context's attributes.
    """

    _, cls_attrs = generate_artifacts(ctx=ctx)
    if "include" in kwargs and "exclude" in kwargs:
        raise RuntimeError("'include' and 'exclude' should not both be present")
    elif "include" in kwargs:
        assert type(kwargs["include"]) is list
        for in_attr in kwargs["include"]:
            if in_attr not in cls_attrs:
                raise RuntimeError(
                    f"argument '{in_attr}' not found in flow task {f.__name__}"
                )
        for attr in cls_attrs:
            if attr not in kwargs["include"]:
                delattr(ctx, attr)
    elif "exclude" in kwargs:
        assert type(kwargs["exclude"]) is list
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
    Optionally saves the current state for the task just executed.

    Args:
        ctx (any): The context to checkpoint.
        parent_func (function): The function that was just executed.
        chkpnt_reserved_words (list, optional): A list of reserved words to exclude from checkpointing. Defaults to ["next", "runtime"].
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
