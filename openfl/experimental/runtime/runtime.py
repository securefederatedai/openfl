# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime module."""


class Runtime:
    def __init__(self):
        """ Interface for runtimes that can run FLSpec flows"""
        pass

class LocalRuntime(Runtime):

   def __init__(self,aggregator=None,collaborators=None):
        """Use remote federated infrastructure to run the flow"""
        super().__init__()    
        self.aggregator = aggregator
        # List of envoys should be iterable, so that a subset can be selected at runtime
        # The envoys is the superset of envoys that can be selected during the experiment
        if collaborators != None:
            self.collaborators = collaborators

   @property
   def aggregator(self):
       return self._aggregator.name

   @aggregator.setter
   def aggregator(self,aggregator):
       self._aggregator = aggregator

   @property
   def collaborators(self):
       """
       Return names of collaborators. Don't give direct access to private attributes
       """
       return list(self._collaborators.keys())

   @collaborators.setter
   def collaborators(self,collaborators):
       self._collaborators = {collaborator.name: collaborator for collaborator in collaborators}

   def __repr__(self):
       return 'LocalRuntime'

class FederatedRuntime(Runtime):
    def __init__(self,aggregator,collaborators=None):
        """Use remote federated infrastructure to run the flow"""
        super().__init__()    
        ray.init()
        self.aggregator = aggregator
        # List of envoys should be iterable, so that a subset can be selected at runtime
        # The envoys is the superset of envoys that can be selected during the experiment
        if collaborators != None:
            self.collaborators = collaborators
        self.envoys = director.get_envoys()

