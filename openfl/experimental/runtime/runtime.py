# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" openfl.experimental.runtime module Runtime class."""

from typing import Callable, List

from openfl.experimental.interface.fl_spec import FLSpec
from openfl.experimental.interface.participants import Aggregator, Collaborator


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
        **kwargs,
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
