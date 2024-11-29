# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Detect criteria for transitions in placement."""


def should_transfer(func, parent_func):
    """Determines if a transfer should occur from collaborator to aggregator.

    Args:
        func (function): The current function.
        parent_func (function): The parent function.

    Returns:
        bool: True if a transfer should occur, False otherwise.
    """
    if collaborator_to_aggregator(func, parent_func):
        return True
    else:
        return False


def aggregator_to_collaborator(func, parent_func):
    """Checks if a transition from aggregator to collaborator is possible.

    Args:
        func (function): The current function.
        parent_func (function): The parent function.

    Returns:
        bool: True if the transition is possible, False otherwise.
    """
    if parent_func.aggregator_step and func.collaborator_step:
        return True
    else:
        return False


def collaborator_to_aggregator(func, parent_func):
    """Checks if a transition from collaborator to aggregator is possible.

    Args:
        func (function): The current function.
        parent_func (function): The parent function.

    Returns:
        bool: True if the transition is possible, False otherwise.
    """
    if parent_func.collaborator_step and func.aggregator_step:
        return True
    else:
        return False
