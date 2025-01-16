# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

pass
import numpy as np
import logging
from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from openfl.experimental.workflow.placement import aggregator, collaborator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TestFlowPrivateAttributes(FLSpec):
    """
    Testflow to validate Aggregator private attributes are not accessible to collaborators
    and vice versa
    """
    __test__ = False # to prevent pytest from trying to discover tests in the class

    @aggregator
    def start(self):
        """
        Flow start.
        """
        log.info("Testing FederatedFlow - Starting Test for accessibility of private attributes")
        self.collaborators = self.runtime.collaborators

        validate_collab_private_attr(self, "test_loader", "start")

        self.exclude_agg_to_agg = 10
        self.include_agg_to_agg = 100
        self.next(self.aggregator_step, exclude=["exclude_agg_to_agg"])

    @aggregator
    def aggregator_step(self):
        """
        Testing whether Agg private attributes are accessible in next agg step.
        Collab private attributes should not be accessible here
        """
        validate_collab_private_attr(self, "test_loader", "aggregator_step")

        self.include_agg_to_collab = 42
        self.exclude_agg_to_collab = 40
        self.next(
            self.collaborator_step_a,
            foreach="collaborators",
            exclude=["exclude_agg_to_collab"],
        )

    @collaborator
    def collaborator_step_a(self):
        """
        Testing whether Collab private attributes are accessible in collab step
        Aggregator private attributes should not be accessible here
        """
        validate_agg_private_attrs(
            self, "train_loader", "test_loader", "collaborator_step_a"
        )

        self.exclude_collab_to_collab = 2
        self.include_collab_to_collab = 22
        self.next(self.collaborator_step_b, exclude=["exclude_collab_to_collab"])

    @collaborator
    def collaborator_step_b(self):
        """
        Testing whether Collab private attributes are accessible in collab step
        Aggregator private attributes should not be accessible here
        """

        validate_agg_private_attrs(
            self, "train_loader", "test_loader", "collaborator_step_b"
        )
        self.exclude_collab_to_agg = 10
        self.include_collab_to_agg = 12
        self.next(self.join, exclude=["exclude_collab_to_agg"])

    @aggregator
    def join(self, inputs):
        """
        Testing whether attributes are excluded from collab to agg
        """
        # Aggregator should only be able to access its own attributes
        assert hasattr(self, "test_loader"), "aggregator_join_aggregator_attributes_missing"

        for idx, collab in enumerate(inputs):
            assert not (hasattr(collab, "train_loader") or hasattr(collab, "test_loader")), \
                f"join_collaborator_attributes_found for Collaborator: {collab}"

        self.next(self.end)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """
        log.info("Testing FederatedFlow - Ending Test for accessibility of private attributes")
        log.info("...Test case passed...")


def validate_collab_private_attr(self, private_attr, step_name):
    """
    Validates the private attributes of the aggregator and collaborators.

    Args:
        private_attr (str): The name of the private attribute to validate.
        step_name (str): The name of the current step in the workflow.

    Raises:
        AssertionError: If the aggregator does not have the specified private attribute.
        AssertionError: If any collaborator's private attributes are accessible.
    """
    # Aggregator should only be able to access its own attributes
    assert hasattr(self, private_attr), f"{step_name}_aggregator_attributes_missing"

    for idx, collab in enumerate(self.collaborators):
        # Collaborator private attributes should not be accessible
        assert not (type(self.collaborators[idx]) is not str or hasattr(self.runtime, "_collaborators") or hasattr(self.runtime, "__collaborators")), \
            f"{step_name}_collaborator_attributes_found for collaborator {collab}"


def validate_agg_private_attrs(self, private_attr_1, private_attr_2, step_name):
    """
    Validates that the collaborator can only access its own private attributes and not the aggregator's attributes.

    Args:
        private_attr_1 (str): The name of the first private attribute to check.
        private_attr_2 (str): The name of the second private attribute to check.
        step_name (str): The name of the current step in the workflow.

    Raises:
        AssertionError: If the collaborator does not have the specified private attributes or if the aggregator's attributes are accessible.
    """
    # Collaborator should only be able to access its own attributes
    assert hasattr(self, private_attr_1) and hasattr(self, private_attr_2), \
        f"{step_name}collab_attributes_not_found"

    assert not hasattr(self.runtime, "_aggregator"), \
        f"{step_name}_aggregator_attributes_found"
