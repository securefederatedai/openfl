# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from openfl.experimental.workflow.interface.fl_spec import FLSpec
from openfl.experimental.workflow.interface.participants import Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from openfl.experimental.workflow.placement.placement import aggregator, collaborator
import numpy as np
pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TestFlowInternalLoop(FLSpec):
    __test__ = False # to prevent pytest from trying to discover tests in the class

    def __init__(self, model=None, optimizer=None, rounds=None, **kwargs):
        super().__init__(**kwargs)
        self.training_rounds = rounds
        self.train_count = 0
        self.end_count = 0

    @aggregator
    def start(self):
        """
        Flow start.
        """
        log.info(f"Testing FederatedFlow - Test for Internal Loops - Round: {self.train_count} of Training Rounds: {self.training_rounds}")
        self.model = np.zeros((10, 10, 10))  # Test model
        self.collaborators = self.runtime.collaborators
        self.next(self.agg_model_mean, foreach="collaborators")

    @collaborator
    def agg_model_mean(self):
        """
        Calculating the mean of the model created in start.
        """
        self.agg_mean_value = np.mean(self.model)
        log.info(f"<Collab>: {self.input} Mean of Agg model: {self.agg_mean_value}")
        self.next(self.collab_model_update)

    @collaborator
    def collab_model_update(self):
        """
        Initializing the model with random numbers.
        """
        log.info(f"<Collab>: {self.input} Initializing the model randomly")
        self.model = np.random.randint(1, len(self.input), (10, 10, 10))
        self.next(self.local_model_mean)

    @collaborator
    def local_model_mean(self):
        """
        Calculating the mean of the model created in train.
        """
        self.local_mean_value = np.mean(self.model)
        log.info(f"<Collab>: {self.input} Local mean: {self.local_mean_value}")
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        """
        Joining inputs from collaborators
        """
        self.agg_mean = sum(input.local_mean_value for input in inputs) / len(inputs)
        log.info(f"Aggregated mean : {self.agg_mean}")
        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        """
        Internally Loop for training rounds
        """
        self.train_count = self.train_count + 1
        if self.training_rounds == self.train_count:
            self.next(self.end)
        else:
            self.next(self.start)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """
        self.end_count = self.end_count + 1
        log.info("This is the end of the flow")
