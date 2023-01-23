# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime module Runtime class."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator, Collaborator, FLSpec
from typing import List
from typing import Callable


class Runtime:
    def __init__(self):
        """
        Base interface for runtimes that can run FLSpec flows

        """
        pass

    @property
    def aggregator(self):
        """Returns name of aggregator"""
        raise NotImplementedError

    @aggregator.setter
    def aggregator(self, aggregator: Aggregator):
        """Set Runtime aggregator"""
        raise NotImplementedError

    @property
    def collaborators(self):
        """
        Return names of collaborators. Don't give direct access to private attributes
        """
        raise NotImplementedError

    @collaborators.setter
    def collaborators(self, collaborators: List[Collaborator]):
        """Set Runtime collaborators"""
        raise NotImplementedError

    def execute_task(
        self,
        flspec_obj: FLSpec,
        f: Callable,
        parent_func: Callable,
        instance_snapshot: List[FLSpec] = [],
        **kwargs
    ):
        """
        Performs the execution of a task as defined by the
        implementation and underlying backend (single_process, ray, etc)

        Args:
            flspec_obj:        Reference to the FLSpec (flow) object. Contains information
                               about task sequence, flow attributes, that are needed to
                               execute a future task
            f:                 The next task to be executed within the flow
            parent_func:       The prior task executed in the flow
            instance_snapshot: A prior FLSpec state that needs to be restored from
                               (i.e. restoring aggregator state after collaborator
                               execution)
        """
        raise NotImplementedError
