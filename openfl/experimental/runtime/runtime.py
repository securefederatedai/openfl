# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime module Runtime class."""


class Runtime:
    def __init__(self):
        """
        Interface for runtimes that can run FLSpec flows

        """
        pass

    @property
    def aggregator(self):
        """Returns name of aggregator"""
        raise NotImplementedError

    @aggregator.setter
    def aggregator(self, aggregator):
        """Set Runtime aggregator"""
        raise NotImplementedError

    @property
    def collaborators(self):
        """
        Return names of collaborators. Don't give direct access to private attributes
        """
        raise NotImplementedError

    @collaborators.setter
    def collaborators(self, collaborators):
        """Set Runtime collaborators"""
        raise NotImplementedError

    def execute_task(
        self, flspec_obj, f, parent_func, instance_snapshot=(), **kwargs
    ):
        """
        Performs the execution of a task as defined by the
        runtime implementation and underlying backend (single_process, ray, etc)
        """
        raise NotImplementedError
