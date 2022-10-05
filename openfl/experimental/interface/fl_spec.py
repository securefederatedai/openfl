# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.flspec module."""


import functools
from multiprocessing import Pool
import inspect
import pickle
import hashlib
import ray
from types import MethodType
from random import randint
from copy import deepcopy, copy
from time import sleep
from openfl.experimental.placement import make_remote
from openfl.experimental.utilities import MetaflowInterface
from openfl.experimental.runtime import LocalRuntime, Runtime
from threading import Lock
import time
import numpy as np
import sys

def should_transfer(func,parent_func):
    if (aggregator_to_collaborator(func,parent_func) or collaborator_to_aggregator(func,parent_func)):
        return True
    else:
        return False

def aggregator_to_collaborator(func,parent_func):
    if (parent_func.aggregator_step and func.collaborator_step):
        return True
    else:
        return False

def collaborator_to_aggregator(func,parent_func):
    if (parent_func.collaborator_step and func.aggregator_step):
        return True
    else:
        return False

final_attributes = []
mutex = Lock()

class FLSpec:

    _clones = []
    _initial_state = None

    def __init__(self,checkpoint=False):
        self._foreach_methods = []
        self._run_id = "%d" % (time.time() * 1e6)
        self._reserved_words = ['next','runtime']
        self._checkpoint = checkpoint
        self._metaflow_interface = MetaflowInterface(self.__class__.__name__)
        self._run_id = self._metaflow_interface.create_run()
        self._runtime = LocalRuntime()

    @classmethod
    def create_clones(cls,instance,names):
        cls._clones = [deepcopy(instance) for name in names] 

    @classmethod
    def reset_clones(cls):
        cls._clones = []

    @classmethod
    def save_initial_state(cls,instance):
        cls._initial_state = deepcopy(instance)

    def run(self):
        # Submit flow to Runtime
        FLSpec.save_initial_state(self)
        if str(self._runtime) == 'LocalRuntime':
            # Setup any necessary ShardDescriptors through the LocalEnvoys
            # Assume that first task always runs on the aggregator
            self._setup_aggregator()
            self._foreach_methods = []
            FLSpec.reset_clones()
            # the start function can just be invoked locally
            if self._checkpoint:
                #self._metaflow_interface = MetaflowInterface(self.__class__.__name__)
                print(f'Created flow {self.__class__.__name__}')
                #self._run_id = self._metaflow_interface.create_run()
            #try:
            self.start()
            #except Exception:
            #    # This makes the flow exit correctly if end has been reached
            #    pass
            for name,attr in final_attributes:
                setattr(self,name,attr)
        elif str(self._runtime) == 'FederatedRuntime':
            # Submit to director
            raise Exception(f'Submission to remote runtime not available yet')
        else:
            raise Exception(f'Runtime not supported')

    def _setup_aggregator(self):
        for name, attr in self.runtime._aggregator.private_attributes.items():
            setattr(self,name,attr)

    @property
    def runtime(self):
        return self._runtime

    @runtime.setter
    def runtime(self, runtime):
        if isinstance(runtime, Runtime):
            self._runtime = runtime
        else:
            raise TypeError(f'{runtime} is not a valid OpenFL Runtime')

    def parse_attrs(self,exclude = []):
        #TODO Persist attributes to local disk, database, object store, etc. here
        cls_attrs = []
        valid_artifacts = []
        for i in inspect.getmembers(self):
            if not hasattr(i[1],'task') and \
               not i[0].startswith('_') and \
               i[0] not in self._reserved_words and \
               i[0] not in exclude:
                if not isinstance(i[1], MethodType):
                    cls_attrs.append(i[0])
                    valid_artifacts.append((i[0],i[1]))
        return cls_attrs, valid_artifacts


    def next(self,f,**kwargs):
        global final_attributes

        if FLSpec._initial_state is None:
            cln = self
        else:
            cln = deepcopy(FLSpec._initial_state)
        parent = inspect.stack()[1][3]
        parent_func = getattr(self,parent)

        #TODO Persist attributes to local disk, database, object store, etc. here
        cls_attrs, valid_artifacts = self.parse_attrs()

        def artifacts_iter():
            # Helper function from metaflow source
            while valid_artifacts:
                var, val = valid_artifacts.pop()
                yield var, val
                
        if self._checkpoint:
            # all objects will be serialized using Metaflow internals
            print(f'Saving data artifacts for {parent_func.__name__}')
            task_id = self._metaflow_interface.create_task(parent_func.__name__)
            self._metaflow_interface.save_artifacts(artifacts_iter(), task_name=parent_func.__name__, task_id=task_id)
            print(f'Saved data artifacts for {parent_func.__name__}')
                                
        if 'include' in kwargs and 'exclude' in kwargs:
            raise RuntimeError("'include' and 'exclude' should not both be present")
        elif 'include' in kwargs:
            assert(type(kwargs['include']) == list)
            for in_attr in kwargs['include']:
                if in_attr not in cls_attrs:
                    raise RuntimeError(f"argument '{in_attr}' not found in flow task {f.__name__}")
            for attr in cls_attrs:
                if attr not in kwargs['include']:
                    delattr(self,attr)
        elif 'exclude' in kwargs:
            assert(type(kwargs['exclude']) == list)
            for in_attr in kwargs['exclude']:
                if in_attr not in cls_attrs:
                    raise RuntimeError(f"argument '{in_attr}' not found in flow task {f.__name__}")
            cls_attrs, valid_artifacts = self.parse_attrs(exclude=kwargs['exclude'])
            for name,attr in artifacts_iter():
                setattr(cln,name,attr)
            cln._foreach_methods = self._foreach_methods

        else:
            # filtering after a foreach may run into problems
            cln = self

        #Reload state for next function if it exists (this may happen remotely)
        if hasattr(self,'input'):
            '{self.__class__.__name__}{self._instance}_{f.__name__}_{self._counter - 1}_'

        if parent in self._foreach_methods:
            self._foreach_methods.append(f.__name__)
            if should_transfer(f,parent_func):
                print(f'Should transfer from {parent_func.__name__} to {f.__name__}')
                self.execute_next = f.__name__
                return

        if aggregator_to_collaborator(f,parent_func):
            print(f'Sending state from aggregator to collaborators')

        elif collaborator_to_aggregator(f,parent_func):
            print(f'Sending state from collaborator to aggregator')


        if 'foreach' in kwargs:
            self._foreach_methods.append(f.__name__)
            values = cln.__getattribute__(kwargs['foreach'])

            if len(FLSpec._clones) == 0:
                FLSpec.create_clones(cln,values) 
            else:
                for clone in FLSpec._clones:
                    cls_attrs, valid_artifacts = self.parse_attrs()              
                    attributes = artifacts_iter()
                    for name,attr in attributes:
                        setattr(clone,name,attr)
                    clone._foreach_methods = self._foreach_methods


            for clone,value in zip(FLSpec._clones,values):
                clone.input = value
                if aggregator_to_collaborator(f,parent_func):
                    #remove private aggregator state
                    for attr in self.runtime._aggregator.private_attributes:
                        self.runtime._aggregator.private_attributes[attr] = getattr(self,attr)
                        if hasattr(clone,attr):
                            delattr(clone,attr)
            # TODO Make this multiprocess
            func = None
            remote_functions = []
            for clone in FLSpec._clones:
                # TODO make task_id a shared value
                for name,attr in self.runtime._collaborators[clone.input].private_attributes.items():
                    setattr(clone,name,attr)
                to_exec = getattr(clone,f.__name__)
                remote_to_exec = make_remote(to_exec)
                # write the clone to the object store
                # ensure clone is getting latest _metaflow_interface
                clone._metaflow_interface = self._metaflow_interface
                clone = ray.put(clone)
                remote_functions.append(remote_to_exec.remote(clone,f.__name__))
            clones = ray.get(remote_functions)
            FLSpec._clones = clones
            for clone in FLSpec._clones:
                func = clone.execute_next
                # This sets up possibility for different collaborators to have custom private attributes
                for attr in self.runtime._collaborators[clone.input].private_attributes:
                    self.runtime._collaborators[clone.input].private_attributes[attr] = getattr(clone,attr)
                    if hasattr(clone,attr):
                        delattr(clone,attr)
            print(f'Next function = {func}')
            g = getattr(self,func)
            # remove private collaborator state
            g(FLSpec._clones)
        else:
            to_exec = getattr(self,f.__name__)
            to_exec()
            if f.__name__ == 'end':
                cls_attrs, valid_artifacts = cln.parse_attrs()              
                final_attributes = artifacts_iter()
                #raise Exception('Exiting flow')

