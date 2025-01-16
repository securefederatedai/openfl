# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import random
from openfl.experimental.workflow.interface.fl_spec import FLSpec
from openfl.experimental.workflow.placement.placement import aggregator, collaborator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TestFlowSubsetCollaborators(FLSpec):
    """
    Testflow to validate working of Subset Collaborators in Federated Flow.
    """
    __test__ = False # to prevent pytest from trying to discover tests in the class

    def __init__(self, random_ints=[], **kwargs) -> None:
        """
        Initialize the SubsetFlow class.

        Args:
            random_ints (list, optional): A list of random integers. Defaults to an empty list.
            **kwargs: Additional keyword arguments passed to the superclass initializer.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.random_ints = random_ints

    @aggregator
    def start(self):
        """
        Starting the flow with random subset of collaborators
        """
        log.info("Testing FederatedFlow - Starting Test for validating Subset of collaborators")
        self.collaborators = self.runtime.collaborators

        # select subset of collaborators
        self.subset_collaborators = self.collaborators[: random.choice(self.random_ints)]

        log.info(f"... Executing flow for {len(self.subset_collaborators)} collaborators out of Total: {len(self.collaborators)}")

        self.next(self.test_valid_collaborators, foreach="subset_collaborators")

    @collaborator
    def test_valid_collaborators(self):
        """
        set the collaborator name
        """
        log.info(f"Print collaborators {self.name}")
        self.collaborator_ran = self.name
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        """
        List of collaborators ran successfully
        """
        log.info("inside join")
        self.collaborators_ran = [i.collaborator_ran for i in inputs]
        self.next(self.end)

    @aggregator
    def end(self):
        """
        End of the flow
        """
        log.info(f"End of the test case {TestFlowSubsetCollaborators.__name__} reached.")
