# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime package LocalRuntime class."""

from copy import deepcopy
import ray
import gc
from openfl.experimental.runtime import Runtime
from openfl.experimental.placement import RayExecutor
from openfl.experimental.utilities import (
    aggregator_to_collaborator,
    generate_artifacts,
    filter_attributes,
    checkpoint,
)


class LocalRuntime(Runtime):
    def __init__(
        self,
        aggregator=None,
        collaborators=None,
        backend="single_process",
        **kwargs,
    ):
        """Use Local infrastructure to run the flow"""
        super().__init__()
        if backend not in ["ray", "single_process"]:
            raise ValueError(
                f"Invalid 'backend' value '{backend}', accepted values are "
                + "'ray', or 'single_process'"
            )
        if backend == "ray":
            if not ray.is_initialized():
                dh = kwargs.get("dashboard_host", "127.0.0.1")
                dp = kwargs.get("dashboard_port", 5252)
                ray.init(dashboard_host=dh, dashboard_port=dp)
        self.backend = backend
        self.aggregator = aggregator
        # List of envoys should be iterable, so that a subset can be selected at runtime
        # The envoys is the superset of envoys that can be selected during the experiment
        if collaborators is not None:
            self.collaborators = collaborators

    @property
    def aggregator(self):
        """Returns name of _aggregator"""
        return self._aggregator.name

    @aggregator.setter
    def aggregator(self, aggregator):
        """Set LocalRuntime _aggregator"""
        self._aggregator = aggregator

    @property
    def collaborators(self):
        """
        Return names of collaborators. Don't give direct access to private attributes
        """
        return list(self._collaborators.keys())

    @collaborators.setter
    def collaborators(self, collaborators):
        """Set LocalRuntime collaborators"""
        self._collaborators = {
            collaborator.name: collaborator for collaborator in collaborators
        }

    def restore_instance_snapshot(self, ctx, instance_snapshot):
        """Restores attributes from backup (in instance snapshot) to ctx"""
        for backup in instance_snapshot:
            artifacts_iter, _ = generate_artifacts(ctx=backup)
            for name, attr in artifacts_iter():
                if not hasattr(ctx, name):
                    setattr(ctx, name, attr)

    def execute_task(
        self, flspec_obj, f, parent_func, instance_snapshot=(), **kwargs
    ):
        """
        Next task execution happens here
        """
        from openfl.experimental.interface import (
            FLSpec,
            final_attributes,
        )

        global final_attributes

        if "foreach" in kwargs:
            flspec_obj._foreach_methods.append(f.__name__)
            selected_collaborators = flspec_obj.__getattribute__(
                kwargs["foreach"]
            )

            for col in selected_collaborators:
                clone = FLSpec._clones[col]
                if (
                    "exclude" in kwargs and hasattr(clone, kwargs["exclude"][0])
                ) or (
                    "include" in kwargs and hasattr(clone, kwargs["include"][0])
                ):
                    filter_attributes(clone, f, **kwargs)
                artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
                for name, attr in artifacts_iter():
                    setattr(clone, name, deepcopy(attr))
                clone._foreach_methods = flspec_obj._foreach_methods

            for col in selected_collaborators:
                clone = FLSpec._clones[col]
                clone.input = col
                if aggregator_to_collaborator(f, parent_func):
                    # remove private aggregator state
                    for attr in self._aggregator.private_attributes:
                        self._aggregator.private_attributes[attr] = getattr(
                            flspec_obj, attr
                        )
                        if hasattr(clone, attr):
                            delattr(clone, attr)

            func = None
            if self.backend == "ray":
                ray_executor = RayExecutor()
            for col in selected_collaborators:
                clone = FLSpec._clones[col]
                for name, attr in self._collaborators[
                    clone.input
                ].private_attributes.items():
                    setattr(clone, name, attr)
                to_exec = getattr(clone, f.__name__)
                # write the clone to the object store
                # ensure clone is getting latest _metaflow_interface
                clone._metaflow_interface = flspec_obj._metaflow_interface
                if self.backend == "ray":
                    clone._runtime = self  # MOD: Runtime()
                    ray_executor.ray_call_put(clone, to_exec)
                else:
                    to_exec()
            if self.backend == "ray":
                clones = ray_executor.get_remote_clones()
                FLSpec._clones.update(
                    {
                        col: obj
                        for col, obj in zip(selected_collaborators, clones)
                    }
                )
                del ray_executor
                del clones
                gc.collect()
            for col in selected_collaborators:
                clone = FLSpec._clones[col]
                func = clone.execute_next
                for attr in self._collaborators[clone.input].private_attributes:
                    if hasattr(clone, attr):
                        self._collaborators[clone.input].private_attributes[
                            attr
                        ] = getattr(clone, attr)
                        delattr(clone, attr)
            # Restore the flspec_obj state if back-up is taken
            self.restore_instance_snapshot(flspec_obj, instance_snapshot)
            del instance_snapshot

            g = getattr(flspec_obj, func)
            # remove private collaborator state
            gc.collect()
            g([FLSpec._clones[col] for col in selected_collaborators])
        else:
            to_exec = getattr(flspec_obj, f.__name__)
            to_exec()
            if f.__name__ == "end":
                checkpoint(flspec_obj, f)
                artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
                final_attributes = artifacts_iter()

    def __repr__(self):
        return "LocalRuntime"
