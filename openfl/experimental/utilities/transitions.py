# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Detect criteria for transitions in placement."""


def should_transfer(func, parent_func):
    if collaborator_to_aggregator(func, parent_func):
        return True
    else:
        return False


def aggregator_to_collaborator(func, parent_func):
    if parent_func.aggregator_step and func.collaborator_step:
        return True
    else:
        return False


def collaborator_to_aggregator(func, parent_func):
    if parent_func.collaborator_step and func.aggregator_step:
        return True
    else:
        return False
