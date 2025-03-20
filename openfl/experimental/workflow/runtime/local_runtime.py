# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.workflow.runtime package LocalRuntime class."""

from __future__ import annotations

import gc
import importlib
import math
import os
from copy import deepcopy
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Type

import ray

from openfl.experimental.workflow.interface.fl_spec import FLSpec
from openfl.experimental.workflow.interface.participants import Aggregator, Collaborator
from openfl.experimental.workflow.runtime.runtime import Runtime
from openfl.experimental.workflow.utilities import (
    ResourcesNotAvailableError,
    aggregator_to_collaborator,
    check_resource_allocation,
    checkpoint,
    filter_attributes,
    generate_artifacts,
    get_number_of_gpus,
)


class RayExecutor:
    """Class for executing tasks using the Ray framework."""

    def __init__(self):
        """Initializes the RayExecutor object."""
        self.__remote_contexts = []

    def ray_call_put(
        self,
        participant: Any,
        ctx: Any,
        f_name: str,
        callback: Callable,
        clones: Optional[Any] = None,
    ) -> None:
        """Execute f_name from inside participant (Aggregator or Collaborator)
        class with the context of clone (ctx).

        Args:
            participant (Any): The participant (Aggregator or Collaborator) to
                execute the function in.
            ctx (Any): The context to execute the function in.
            f_name (str): The name of the function to execute.
            callback (Callable): The callback to execute after the function.
            clones (Optional[Any], optional): The clones to use in the
                function. Defaults to None.
        """
        if clones is not None:
            self.__remote_contexts.append(
                participant.execute_func.remote(ctx, f_name, callback, clones)
            )
        else:
            self.__remote_contexts.append(participant.execute_func.remote(ctx, f_name, callback))

    def ray_call_get(self) -> List[Any]:
        """Get remote clones and delete ray references of clone (ctx) and,
        reclaim memory.

        Returns:
            List[Any]: The list of remote clones.
        """
        clones = ray.get(self.__remote_contexts)
        del self.__remote_contexts
        self.__remote_contexts = []

        return clones


def ray_group_assign(collaborators, num_actors=1):
    """Assigns collaborators to resource groups which share a CUDA context.

    Args:
        collaborators (list): The list of collaborators.
        num_actors (int, optional): Number of actors to distribute
            collaborators to. Defaults to 1.

    Returns:
        list: A list of GroupMember instances.
    """

    class GroupMember:
        """A utility class that manages the collaborator and its group.

        This class maintains compatibility with runtime execution by assigning
        attributes for each function in the Collaborator interface in
        conjunction with RemoteHelper.
        """

        def __init__(self, collaborator_actor, collaborator):
            """Initializes a new instance of the GroupMember class.

            Args:
                collaborator_actor: The collaborator actor.
                collaborator: The collaborator.
            """

            all_methods = [
                method for method in dir(Collaborator) if callable(getattr(Collaborator, method))
            ]
            external_methods = [method for method in all_methods if (method[0] != "_")]
            self.collaborator_actor = collaborator_actor
            self.collaborator = collaborator
            for method in external_methods:
                setattr(
                    self,
                    method,
                    RemoteHelper(self.collaborator_actor, self.collaborator, method),
                )

    class RemoteHelper:
        """A utility class to maintain compatibility with RayExecutor.

        This class returns a lambda function that uses
        collaborator_actor.execute_from_col to run a given function from the
        given collaborator.
        """

        # once ray_grouped replaces the current ray runtime this class can be
        # replaced with a function that returns the lambda function, using a
        # function is necessary because this is used in setting multiple
        # functions in a loop and lambda takes the reference to self.f_name and
        # not the value so we need to change scope to avoid self.f_name from
        # changing as the loop  progresses
        def __init__(self, collaborator_actor, collaborator, f_name) -> None:
            """Initializes a new instance of the RemoteHelper class.

            Args:
                collaborator_actor: The collaborator actor.
                collaborator: The collaborator.
                f_name (str): The name of the function.
            """
            self.f_name = f_name
            self.collaborator_actor = collaborator_actor
            self.collaborator = collaborator
            self.f = lambda *args, **kwargs: self.collaborator_actor.execute_from_col.remote(
                self.collaborator, self.f_name, *args, **kwargs
            )

        def remote(self, *args, **kwargs):
            """Executes the function with the given arguments and keyword
            arguments.

            Args:
                *args: The arguments to pass to the function.
                **kwargs: The keyword arguments to pass to the function.

            Returns:
                The result of the function execution.
            """
            return self.f(*args, *kwargs)

    collaborator_ray_refs = []
    collaborators_per_group = math.ceil(len(collaborators) / num_actors)
    times_called = 0
    # logic to sort collaborators by gpus, if collaborators have the same
    # number of gpu then they are sorted by cpu
    cpu_magnitude = len(str(abs(max([i.num_cpus for i in collaborators]))))
    min_gpu = min([i.num_gpus for i in collaborators])
    min_gpu = max(min_gpu, 0.0001)
    collaborators_sorted_by_gpucpu = sorted(
        collaborators,
        key=lambda x: x.num_gpus / min_gpu * 10**cpu_magnitude + x.num_cpus,
    )
    initializations = []
    collaborator_actor = None

    for collaborator in collaborators_sorted_by_gpucpu:
        # initialize actor group
        if times_called % collaborators_per_group == 0:
            max_num_cpus = max(
                [
                    i.num_cpus
                    for i in collaborators_sorted_by_gpucpu[
                        times_called : times_called + collaborators_per_group
                    ]
                ]
            )
            max_num_gpus = max(
                [
                    i.num_gpus
                    for i in collaborators_sorted_by_gpucpu[
                        times_called : times_called + collaborators_per_group
                    ]
                ]
            )
            print(f"creating actor with {max_num_cpus}, {max_num_gpus}")
            collaborator_actor = (
                ray.remote(RayGroup)
                .options(
                    num_cpus=max_num_cpus, num_gpus=max_num_gpus
                )  # max_concurrency=max_concurrency)
                .remote()
            )

        if collaborator_actor is None:
            raise ValueError("collaborator_actor has not been initialized.")

        # add collaborator to actor group
        initializations.append(collaborator_actor.append.remote(collaborator))

        times_called += 1

        # append GroupMember to output list
        collaborator_ray_refs.append(GroupMember(collaborator_actor, collaborator.get_name()))
    # Wait for all collaborators to be created on actors
    ray.get(initializations)

    return collaborator_ray_refs


class RayGroup:
    """A Ray actor that manages a group of collaborators.

    This class allows for the execution of functions from a specified
    collaborator using the execute_from_col method. The collaborators are
    stored in a dictionary where the key is the collaborator's name.
    """

    def __init__(self):
        """Initializes a new instance of the RayGroup class."""
        self.collaborators = {}

    def append(
        self,
        collaborator: Collaborator,
    ):
        """Appends a new collaborator to the group.

        Args:
            name (str): The name of the collaborator.
            private_attributes_callable (Callable): A callable that sets the
                private attributes of the collaborator.
            **kwargs: Additional keyword arguments.
        """

        if collaborator.private_attributes_callable is not None:
            self.collaborators[collaborator.name] = Collaborator(
                name=collaborator.name,
                private_attributes_callable=collaborator.private_attributes_callable,
                **collaborator.kwargs,
            )
        elif collaborator.private_attributes is not None:
            self.collaborators[collaborator.name] = Collaborator(
                name=collaborator.name,
                **collaborator.kwargs,
            )
            self.collaborators[collaborator.name].initialize_private_attributes(
                collaborator.private_attributes
            )

    def execute_from_col(self, name, internal_f_name, *args, **kwargs):
        """Executes a function from a specified collaborator.

        Args:
            name (str): The name of the collaborator.
            internal_f_name (str): The name of the function to execute.
            *args: Additional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            The result of the function execution.
        """
        f = getattr(self.collaborators[name], internal_f_name)
        return f(*args, **kwargs)

    def get_collaborator(self, name):
        """Retrieves a collaborator from the group by name.

        Args:
            name (str): The name of the collaborator.

        Returns:
            The collaborator instance.
        """
        return self.collaborators[name]


class LocalRuntime(Runtime):
    """Class for a local runtime, derived from the Runtime class.

    Attributes:
        aggregator (Type[Aggregator]): The aggregator participant.
        __collaborators (dict): The collaborators, stored as a dictionary of
            names to participants.
        backend (str): The backend that will execute the tasks.
    """

    def __init__(
        self,
        aggregator: Dict = None,
        collaborators: Dict = None,
        backend: str = "single_process",
        **kwargs,
    ) -> None:
        """Initializes the LocalRuntime object to run the flow on a single
        node, with an optional aggregator, an optional list of collaborators,
        an optional backend, and additional keyword arguments.

        Args:
            aggregator (Type[Aggregator], optional): The aggregator instance
                that holds private attributes.
            collaborators (List[Type[Collaborator]], optional): A list of
                collaborators; each with their own private attributes.
            backend (str, optional): The backend that will execute the tasks.
            Defaults to "single_process".
                Available options are:
                - 'single_process': (default) Executes every task within the
                  same process.
                - 'ray': Executes tasks using the Ray library. We use ray
                  actors called RayGroups to runs tasks in their own isolated
                  process. Each participant is distributed into a ray group.
                  The RayGroups run concurrently while participants in the
                  group run serially.
                  The default is 1 RayGroup and can be changed by using the
                  num_actors=1 kwarg. By using more RayGroups more concurrency
                  is allowed with the trade off being that each RayGroup has
                  extra memory overhead in the form of extra CUDA CONTEXTS.

                  Also the ray runtime supports GPU isolation using Ray's
                  'num_gpus' argument, which can be passed in through the
                  collaborator placement decorator.

        Raises:
            ValueError: If the provided backend value is not 'ray' or
                'single_process'.

        Example:
            @collaborator(num_gpus=1)
            def some_collaborator_task(self):
                # Task implementation
            ...

            By selecting num_gpus=1, the task is guaranteed exclusive GPU
            access. If the system has one GPU, collaborator tasks will run
            sequentially.
        """
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

            self.num_actors = kwargs.get("num_actors", 1)
        self.backend = backend
        self.logger = getLogger(__name__)
        if aggregator is not None:
            self.aggregator = self.__get_aggregator_object(aggregator)

        if collaborators is not None:
            self.collaborators = self.__get_collaborator_object(collaborators)

    def __get_aggregator_object(self, aggregator: Type[Aggregator]) -> Any:
        """Get aggregator object based on localruntime backend.

        If the backend is 'single_process', it returns the aggregator directly.
        If the backend is 'ray', it creates a Ray actor for the aggregator
        with the specified resources.

        Args:
            aggregator (Type[Aggregator]): The aggregator class to instantiate.

        Returns:
            Any: The aggregator object or a reference to the Ray actor
                representing the aggregator.

        Raises:
            ResourcesNotAvailableError: If the requested resources exceed the
                available resources.
        """

        if aggregator.private_attributes and aggregator.private_attributes_callable:
            self.logger.warning(
                "Warning: Aggregator private attributes "
                + "will be initialized via callable and "
                + "attributes via aggregator.private_attributes "
                + "will be ignored"
            )

        if self.backend == "single_process":
            return aggregator

        total_available_cpus = os.cpu_count()
        total_available_gpus = get_number_of_gpus()

        agg_cpus = aggregator.num_cpus
        agg_gpus = aggregator.num_gpus

        if agg_gpus > 0:
            check_resource_allocation(
                total_available_gpus,
                {aggregator.get_name(): agg_gpus},
            )

        if total_available_gpus < agg_gpus:
            raise ResourcesNotAvailableError(
                f"cannot assign more than available GPUs \
                    ({agg_gpus} < {total_available_gpus})."
            )
        if total_available_cpus < agg_cpus:
            raise ResourcesNotAvailableError(
                f"cannot assign more than available CPUs \
                    ({agg_cpus} < {total_available_cpus})."
            )

        interface_module = importlib.import_module("openfl.experimental.workflow.interface")
        aggregator_class = interface_module.Aggregator

        aggregator_actor = ray.remote(aggregator_class).options(
            num_cpus=agg_cpus, num_gpus=agg_gpus
        )

        if aggregator.private_attributes_callable is not None:
            aggregator_actor_ref = aggregator_actor.remote(
                name=aggregator.get_name(),
                private_attributes_callable=aggregator.private_attributes_callable,
                **aggregator.kwargs,
            )
        elif aggregator.private_attributes is not None:
            aggregator_actor_ref = aggregator_actor.remote(
                name=aggregator.get_name(),
                **aggregator.kwargs,
            )
            aggregator_actor_ref.initialize_private_attributes.remote(aggregator.private_attributes)

        return aggregator_actor_ref

    def __get_collaborator_object(self, collaborators: List) -> Any:
        """Get collaborator object based on localruntime backend.

        If the backend is 'single_process', it returns the list of
        collaborators directly.
        If the backend is 'ray', it assigns collaborators to Ray actors using
        the ray_group_assign function.

        Args:
            collaborators (List[Type[Collaborator]]): The list of collaborator
                classes to instantiate.

        Returns:
            Any: The list of collaborator objects or a list of references to
                the Ray actors representing the collaborators.

        Raises:
            ResourcesNotAvailableError: If the requested resources exceed the
                available resources.
        """
        for collab in collaborators:
            if collab.private_attributes and collab.private_attributes_callable:
                self.logger.warning(
                    f"Warning: Collaborator {collab.name} private attributes "
                    + "will be initialized via callable and "
                    + "attributes via collaborator.private_attributes "
                    + "will be ignored"
                )

        if self.backend == "single_process":
            return collaborators

        total_available_cpus = os.cpu_count()
        total_required_cpus = sum([collaborator.num_cpus for collaborator in collaborators])
        if total_available_cpus < total_required_cpus:
            raise ResourcesNotAvailableError(
                f"cannot assign more than available CPUs \
                    ({total_required_cpus} < {total_available_cpus})."
            )

        if self.backend == "ray":
            collaborator_ray_refs = ray_group_assign(collaborators, num_actors=self.num_actors)
            return collaborator_ray_refs

    @property
    def aggregator(self) -> str:
        """Gets the name of the aggregator.

        Returns:
            str: The name of the aggregator.
        """
        return self._aggregator.name

    @aggregator.setter
    def aggregator(self, aggregator: Type[Aggregator]):
        """Set LocalRuntime _aggregator.

        Args:
            aggregator (Type[Aggregator]): The aggregator to be set.
        """
        self._aggregator = aggregator

    @property
    def collaborators(self) -> List[str]:
        """Return names of collaborators. Don't give direct access to private
        attributes.

        Returns:
            List[str]: The names of the collaborators.
        """
        return list(self.__collaborators.keys())

    @collaborators.setter
    def collaborators(self, collaborators: List[Type[Collaborator]]):
        """Set LocalRuntime collaborators.

        Args:
            collaborators (List[Type[Collaborator]]): The collaborators to be
                set.
        """
        if self.backend == "single_process":

            def get_collab_name(collab):
                return collab.get_name()

        else:

            def get_collab_name(collab):
                return ray.get(collab.get_name.remote())

        self.__collaborators = {
            get_collab_name(collaborator): collaborator for collaborator in collaborators
        }

    def get_collaborator_kwargs(self, collaborator_name: str):
        """Returns kwargs of collaborator.

        Args:
            collaborator_name: Collaborator name for which kwargs is to be
                returned

        Returns:
            kwargs: Collaborator private_attributes_callable function name, and
             arguments required to call it.
        """
        collab = self.__collaborators[collaborator_name]
        kwargs = {}
        if hasattr(collab, "private_attributes_callable"):
            if collab.private_attributes_callable is not None:
                kwargs.update(collab.kwargs)
                kwargs["private_attributes_callable"] = collab.private_attributes_callable.__name__

        return kwargs

    def initialize_aggregator(self):
        """Initialize aggregator private attributes."""
        if self.backend == "single_process":
            self._aggregator.initialize_private_attributes()
        else:
            ray.get(self._aggregator.initialize_private_attributes.remote())

    def initialize_collaborators(self):
        """Initialize collaborator private attributes."""
        if self.backend == "single_process":

            def init_private_attrs(collab):
                return collab.initialize_private_attributes()

        else:

            def init_private_attrs(collab):
                return ray.get(collab.initialize_private_attributes.remote())

        for collaborator in self.__collaborators.values():
            init_private_attrs(collaborator)

    def restore_instance_snapshot(self, ctx: Type[FLSpec], instance_snapshot: List[Type[FLSpec]]):
        """Restores attributes from backup (in instance snapshot) to context
        (ctx).

        Args:
            ctx (Type[FLSpec]): The context to restore the snapshot to.
            instance_snapshot (List[Type[FLSpec]]): The snapshot of the
                instance to be restored.
        """
        for backup in instance_snapshot:
            artifacts_iter, _ = generate_artifacts(ctx=backup)
            for name, attr in artifacts_iter():
                if not hasattr(ctx, name):
                    setattr(ctx, name, attr)

    def execute_agg_steps(self, ctx: Any, f_name: str, clones: Optional[Any] = None):
        """
        Execute aggregator steps until at transition point.

        Args:
            ctx (Any): The context in which the function is executed.
            f_name (str): The name of the function to be executed.
            clones (Optional[Any], optional): Clones if any. Defaults to None.
        """
        if clones is not None:
            f = getattr(ctx, f_name)
            f(clones)
        else:
            not_at_transition_point = True
            while not_at_transition_point:
                f = getattr(ctx, f_name)
                f()

                f, parent_func = ctx.execute_task_args[:2]
                if aggregator_to_collaborator(f, parent_func) or f.__name__ == "end":
                    not_at_transition_point = False

                f_name = f.__name__

    def execute_collab_steps(self, ctx: Any, f_name: str):
        """Execute collaborator steps until at transition point.

        Args:
            ctx (Any): The context in which the function is executed.
            f_name (str): The name of the function to be executed.
        """
        not_at_transition_point = True
        while not_at_transition_point:
            f = getattr(ctx, f_name)
            f()

            f, parent_func = ctx.execute_task_args[:2]
            if ctx._is_at_transition_point(f, parent_func):
                not_at_transition_point = False

            f_name = f.__name__

    def execute_task(self, flspec_obj: Type[FLSpec], f: Callable, **kwargs):
        """Defines which function to be executed based on name and kwargs.

        Updates the arguments and executes until end is not reached.

        Args:
            flspec_obj: Reference to the FLSpec (flow) object. Contains
                information about task sequence, flow attributes.
            f: The next task to be executed within the flow.

        Returns:
            artifacts_iter: Iterator with updated sequence of values
        """
        parent_func = None
        instance_snapshot = None
        self.join_step = False

        while f.__name__ != "end":
            if "foreach" in kwargs:
                flspec_obj = self.execute_collab_task(
                    flspec_obj, f, parent_func, instance_snapshot, **kwargs
                )
            else:
                flspec_obj = self.execute_agg_task(flspec_obj, f)
            f, parent_func, instance_snapshot, kwargs = flspec_obj.execute_task_args
        else:
            flspec_obj = self.execute_agg_task(flspec_obj, f)
            f = flspec_obj.execute_task_args[0]

            checkpoint(flspec_obj, f)
            artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
            return artifacts_iter()

    def execute_agg_task(self, flspec_obj, f):
        """Performs execution of aggregator task.

        Args:
            flspec_obj: Reference to the FLSpec (flow) object.
            f: The task to be executed within the flow.

        Returns:
            flspec_obj: updated FLSpec (flow) object.
        """

        aggregator = self._aggregator
        clones = None

        if self.join_step:
            clones = [FLSpec._clones[col] for col in self.selected_collaborators]
            self.join_step = False

        if self.backend == "ray":
            ray_executor = RayExecutor()
            ray_executor.ray_call_put(
                aggregator,
                flspec_obj,
                f.__name__,
                self.execute_agg_steps,
                clones,
            )
            flspec_obj = ray_executor.ray_call_get()[0]
            del ray_executor
        else:
            aggregator.execute_func(flspec_obj, f.__name__, self.execute_agg_steps, clones)

        gc.collect()
        return flspec_obj

    def execute_collab_task(self, flspec_obj, f, parent_func, instance_snapshot, **kwargs):
        """
        Performs
            1. Filter include/exclude
            2. Set runtime, collab private attributes , metaflow_interface
            3. Execution of all collaborator for each task
            4. Remove collaborator private attributes
            5. Execute the next function after transition

        Args:
            flspec_obj: Reference to the FLSpec (flow) object.
            f: The task to be executed within the flow.
            parent_func: The prior task executed in the flow.
            instance_snapshot: A prior FLSpec state that needs to be restored.

        Returns:
            flspec_obj: updated FLSpec (flow) object
        """

        flspec_obj._foreach_methods.append(f.__name__)
        selected_collaborators = getattr(flspec_obj, kwargs["foreach"])
        self.selected_collaborators = selected_collaborators

        # filter exclude/include attributes for clone
        self.filter_exclude_include(flspec_obj, f, selected_collaborators, **kwargs)

        if self.backend == "ray":
            ray_executor = RayExecutor()
        # set runtime,collab private attributes and metaflowinterface
        for col in selected_collaborators:
            clone = FLSpec._clones[col]
            # Set new LocalRuntime for clone as it is required
            # new runtime object will not contain private attributes of
            # aggregator or other collaborators
            clone.runtime = LocalRuntime(backend="single_process")

            # write the clone to the object store
            # ensure clone is getting latest _metaflow_interface
            clone._metaflow_interface = flspec_obj._metaflow_interface

        for collab_name in selected_collaborators:
            clone = FLSpec._clones[collab_name]
            collaborator = self.__collaborators[collab_name]

            if self.backend == "ray":
                ray_executor.ray_call_put(
                    collaborator, clone, f.__name__, self.execute_collab_steps
                )
            else:
                collaborator.execute_func(clone, f.__name__, self.execute_collab_steps)

        if self.backend == "ray":
            clones = ray_executor.ray_call_get()
            FLSpec._clones.update(zip(selected_collaborators, clones))
            clone = clones[0]
            del clones

        flspec_obj.execute_task_args = clone.execute_task_args

        # Restore the flspec_obj state if back-up is taken
        self.restore_instance_snapshot(flspec_obj, instance_snapshot)
        del instance_snapshot

        gc.collect()
        # Setting the join_step to indicate to aggregator to collect clones
        self.join_step = True
        return flspec_obj

    def filter_exclude_include(self, flspec_obj, f, selected_collaborators, **kwargs):
        """
        This function filters exclude/include attributes
        Args:
            flspec_obj: Reference to the FLSpec (flow) object.
            f: The task to be executed within the flow.
            selected_collaborators: all collaborators.
        """

        for col in selected_collaborators:
            clone = FLSpec._clones[col]
            clone.input = col
            if ("exclude" in kwargs and hasattr(clone, kwargs["exclude"][0])) or (
                "include" in kwargs and hasattr(clone, kwargs["include"][0])
            ):
                filter_attributes(clone, f, **kwargs)
            artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
            for name, attr in artifacts_iter():
                setattr(clone, name, deepcopy(attr))
            clone._foreach_methods = flspec_obj._foreach_methods

    def __repr__(self):
        """Returns the string representation of the LocalRuntime object.

        Returns:
            str: The string representation of the LocalRuntime object.
        """
        return "LocalRuntime"
