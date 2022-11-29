# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.interface.flspec module."""


import functools
from multiprocessing import Pool
import inspect
import pickle
import hashlib
import ray
from types import MethodType
from random import randint
from copy import deepcopy
from time import sleep
from openfl.experimental.placement import make_remote, ray_call_put
from openfl.experimental.utilities import MetaflowInterface, SerializationException, aggregator_to_collaborator, collaborator_to_aggregator, should_transfer
from openfl.experimental.runtime import LocalRuntime, Runtime
from types import MethodType
from threading import Lock
import time
import numpy as np
import sys

final_attributes = []

class FLSpec:

    _clones = []
    _initial_state = None

    def __init__(self, checkpoint=False):
        self._foreach_methods = []
        self._reserved_words = ['next', 'runtime']
        self._checkpoint = checkpoint

    @classmethod
    def create_clones(cls,instance,names):
        cls._clones = {name: deepcopy(instance) for name in names} 

    @classmethod
    def reset_clones(cls):
        cls._clones = []

    @classmethod
    def save_initial_state(cls, instance):
        cls._initial_state = deepcopy(instance)

    def run(self):
        # Submit flow to Runtime
        FLSpec.save_initial_state(self)
        self._metaflow_interface = MetaflowInterface(self.__class__, self.runtime.backend)
        self._run_id = self._metaflow_interface.create_run()
        if str(self._runtime) == 'LocalRuntime':
            # Setup any necessary ShardDescriptors through the LocalEnvoys
            # Assume that first task always runs on the aggregator
            self._setup_aggregator()
            self._foreach_methods = []
            FLSpec.reset_clones()
            FLSpec.create_clones(self,self.runtime.collaborators)
            # the start function can just be invoked locally
            if self._checkpoint:
                print(f'Created flow {self.__class__.__name__}')
            try:
                self.start()
            except Exception as e:
                if "cannot pickle" in str(e) or "Failed to unpickle" in str(e):
                    msg = "\nA serialization error was encountered that could not" \
                        "\nbe handled by the ray backend." \
                        "\nTry rerunning the flow without ray as follows:\n" \
                        "\nLocalRuntime(...,backend='single_process')\n" \
                        "\n or for more information about the original error," \
                        "\nPlease see the official Ray documentation" \
                        "\nhttps://docs.ray.io/en/latest/ray-core/objects/serialization.html"
                    raise SerializationException(str(e)+msg)
                else:
                    raise e
            for name, attr in final_attributes:
                setattr(self, name, attr)
        elif str(self._runtime) == 'FederatedRuntime':
            # Submit to director
            raise Exception(f'Submission to remote runtime not available yet')
        else:
            raise Exception(f'Runtime not supported')

    def _setup_aggregator(self):
        for name, attr in self.runtime._aggregator.private_attributes.items():
            setattr(self, name, attr)

    @property
    def runtime(self):
        return self._runtime

    @runtime.setter
    def runtime(self, runtime):
        if isinstance(runtime, Runtime):
            self._runtime = runtime
        else:
            raise TypeError(f'{runtime} is not a valid OpenFL Runtime')

    def parse_attrs(self, exclude=[]):
        # TODO Persist attributes to local disk, database, object store, etc. here
        cls_attrs = []
        valid_artifacts = []
        for i in inspect.getmembers(self):
            if not hasattr(i[1], 'task') and \
               not i[0].startswith('_') and \
               i[0] not in self._reserved_words and \
               i[0] not in exclude:
                if not isinstance(i[1], MethodType):
                    cls_attrs.append(i[0])
                    valid_artifacts.append((i[0], i[1]))
        return cls_attrs, valid_artifacts

    def filter_attributes(self, cln, **kwargs):
        """
        Filter out explicitly included / excluded attributes from the next task
        in the flow.
        """

        artifacts_iter, cls_attrs = self.generate_artifacts()
        if 'include' in kwargs and 'exclude' in kwargs:
            raise RuntimeError(
                "'include' and 'exclude' should not both be present")
        elif 'include' in kwargs:
            assert (type(kwargs['include']) == list)
            for in_attr in kwargs['include']:
                if in_attr not in cls_attrs:
                    raise RuntimeError(
                        f"argument '{in_attr}' not found in flow task {f.__name__}")
            for attr in cls_attrs:
                if attr not in kwargs['include']:
                    delattr(self, attr)
        elif 'exclude' in kwargs:
            assert (type(kwargs['exclude']) == list)
            for in_attr in kwargs['exclude']:
                if in_attr not in cls_attrs:
                    raise RuntimeError(
                        f"argument '{in_attr}' not found in flow task {f.__name__}")
            cls_attrs, valid_artifacts = self.parse_attrs(
                exclude=kwargs['exclude'])
            for name, attr in artifacts_iter():
                setattr(cln, name, attr)
            cln._foreach_methods = self._foreach_methods

        else:
            # filtering after a foreach may run into problems
            cln = self

    def checkpoint(self, parent_func):
        """
        [Optionally] save current state for the task just executed task
        """

        # Extract the stdout & stderr from the buffer
        # NOTE: Any prints in this method before this line will be recorded as step output/error
        step_stdout, step_stderr = parent_func._stream_buffer.get_stdstream()

        if self._checkpoint:
            # all objects will be serialized using Metaflow interface
            print(f'Saving data artifacts for {parent_func.__name__}')
            artifacts_iter, _ = self.generate_artifacts()
            task_id = self._metaflow_interface.create_task(
                parent_func.__name__)
            self._metaflow_interface.save_artifacts(
                artifacts_iter(), task_name=parent_func.__name__, task_id=task_id, buffer_out=step_stdout, buffer_err=step_stderr)
            print(f'Saved data artifacts for {parent_func.__name__}')

    def create_clone(self):
        """
        Create a copy of the current state to modify for the next task
        in the flow while leaving the current object untouched.
        """
        if FLSpec._initial_state is None:
            cln = self
        else:
            runtime = FLSpec._initial_state.runtime
            FLSpec._initial_state.runtime = Runtime()
            cln = deepcopy(FLSpec._initial_state)
            FLSpec._initial_state.runtime = runtime
        return cln

    def generate_artifacts(self):

        cls_attrs, valid_artifacts = self.parse_attrs()

        def artifacts_iter():
            # Helper function from metaflow source
            while valid_artifacts:
                var, val = valid_artifacts.pop()
                yield var, val
        return artifacts_iter, cls_attrs

    def at_transition_point(self, f, parent_func):
        """
        Has the collaborator finished its current sequence?
        """
        if parent_func.__name__ in self._foreach_methods:
            self._foreach_methods.append(f.__name__)
            if should_transfer(f, parent_func):
                print(
                    f'Should transfer from {parent_func.__name__} to {f.__name__}')
                self.execute_next = f.__name__
                return True
        return False

    def display_transition_logs(self, f, parent_func):
        if aggregator_to_collaborator(f, parent_func):
            print(f'Sending state from aggregator to collaborators')

        elif collaborator_to_aggregator(f, parent_func):
            print(f'Sending state from collaborator to aggregator')

    def execute_task(self, cln, f, parent_func, **kwargs):
        """
        Next task execution happens here
        """

        global final_attributes

        if 'foreach' in kwargs:
            self._foreach_methods.append(f.__name__)
            selected_collaborators = cln.__getattribute__(kwargs['foreach'])

            for col in selected_collaborators:
                clone = FLSpec._clones[col]
                artifacts_iter, _ = cln.generate_artifacts()
                attributes = artifacts_iter()
                for name,attr in attributes:
                    setattr(clone,name,deepcopy(attr))
                clone._foreach_methods = self._foreach_methods

            for col in selected_collaborators:
                clone = FLSpec._clones[col]
                clone.input = col
                if aggregator_to_collaborator(f,parent_func):
                    #remove private aggregator state
                    for attr in self.runtime._aggregator.private_attributes:
                        self.runtime._aggregator.private_attributes[attr] = getattr(self,attr)
                        if hasattr(clone,attr):
                            delattr(clone,attr)

            func = None
            remote_functions = []
            for col in selected_collaborators:
                # TODO make task_id a shared value
                clone = FLSpec._clones[col]
                for name,attr in self.runtime._collaborators[clone.input].private_attributes.items():
                    setattr(clone,name,attr)
                to_exec = getattr(clone,f.__name__)
                # write the clone to the object store
                # ensure clone is getting latest _metaflow_interface
                clone._metaflow_interface = self._metaflow_interface
                if self._runtime.backend == "ray":     
                    remote_functions.append(ray_call_put(clone, to_exec))
                else:
                    to_exec()
            if self._runtime.backend == "ray":
                FLSpec._clones.update({col: obj for col,obj in \
                    zip(selected_collaborators, ray.get(remote_functions))})
            for col in selected_collaborators: 
                clone = FLSpec._clones[col]
                func = clone.execute_next
                # This sets up possibility for different collaborators to have custom private attributes
                for attr in self.runtime._collaborators[clone.input].private_attributes:
                    self.runtime._collaborators[clone.input].private_attributes[attr] = getattr(
                        clone, attr)
                    if hasattr(clone, attr):
                        delattr(clone, attr)
            g = getattr(self, func)
            # remove private collaborator state
            g([FLSpec._clones[col] for col in selected_collaborators])
        else:
            to_exec = getattr(self, f.__name__)
            to_exec()
            if f.__name__ == 'end':
                artifacts_iter, _ = self.generate_artifacts()
                final_attributes = artifacts_iter()

    def next(self, f, **kwargs):
        """
        Next task in the flow to execute
        """

        # Make a local clone to isolate state for the next task
        cln = self.create_clone()

        # Get the name and reference to the calling function
        parent = inspect.stack()[1][3]
        parent_func = getattr(self, parent)

        # Checkpoint current attributes (if checkpoint==True)
        self.checkpoint(parent_func)

        # Remove included / excluded attributes from next task
        self.filter_attributes(cln, **kwargs)
        
        if self.at_transition_point(f, parent_func):
            # Collaborator is done executing for now
            return

        self.display_transition_logs(f, parent_func)

        self.execute_task(cln, f, parent_func, **kwargs)


