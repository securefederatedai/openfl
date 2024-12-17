# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator

log = logging.getLogger(__name__)

class TestFlowExclude(FLSpec):
    """
    Testflow to validate exclude functionality in Federated Flow
    """

    @aggregator
    def start(self):
        """
        Flow start.
        """
        log.info("Testing WorkFlow - Starting Test for Exclude Attributes")
        self.collaborators = self.runtime.collaborators

        self.exclude_agg_to_agg = 10
        self.include_agg_to_agg = 100
        self.next(self.test_exclude_agg_to_agg, exclude=["exclude_agg_to_agg"])

    @aggregator
    def test_exclude_agg_to_agg(self):
        """
        Testing whether attributes are excluded from agg to agg
        """
        assert hasattr(self, "include_agg_to_agg") is True, "include_agg_to_agg attribute missing"
        assert hasattr(self, "exclude_agg_to_agg") is False, "exclude_agg_to_agg attribute should be excluded"
        log.info("Exclude test passed in test_exclude_agg_to_agg")

        self.exclude_agg_to_collab = 20
        self.include_agg_to_collab = 100
        self.next(
            self.test_exclude_agg_to_collab,
            foreach="collaborators",
            exclude=["exclude_agg_to_collab"],
        )

    @collaborator
    def test_exclude_agg_to_collab(self):
        """
        Testing whether attributes are excluded from agg to collab
        """

        assert hasattr(self, "include_agg_to_agg") is True, "include_agg_to_agg attribute missing"
        assert hasattr(self, "include_agg_to_collab") is True, "include_agg_to_collab attribute missing"
        assert hasattr(self, "exclude_agg_to_agg") is False, "exclude_agg_to_agg attribute should be excluded"
        assert hasattr(self, "exclude_agg_to_collab") is False, "exclude_agg_to_collab attribute should be excluded"
        log.info("Exclude test passed in test_exclude_agg_to_collab")

        self.exclude_collab_to_collab = 10
        self.include_collab_to_collab = 44
        self.next(
            self.test_exclude_collab_to_collab,
            exclude=["exclude_collab_to_collab"],
        )

    @collaborator
    def test_exclude_collab_to_collab(self):
        """
        Testing whether attributes are excluded from collab to collab
        """

        assert hasattr(self, "include_agg_to_agg") is True, "include_agg_to_agg attribute missing"
        assert hasattr(self, "include_agg_to_collab") is True, "include_agg_to_collab attribute missing"
        assert hasattr(self, "include_collab_to_collab") is True, "include_collab_to_collab attribute missing"
        assert hasattr(self, "exclude_agg_to_agg") is False, "exclude_agg_to_agg attribute should be excluded"
        assert hasattr(self, "exclude_agg_to_collab") is False, "exclude_agg_to_collab attribute should be excluded"
        assert hasattr(self, "exclude_collab_to_collab") is False, "exclude_collab_to_collab attribute should be excluded"
        log.info("Exclude test passed in test_exclude_collab_to_collab")

        self.exclude_collab_to_agg = 20
        self.include_collab_to_agg = 56
        self.next(self.join, exclude=["exclude_collab_to_agg"])

    @aggregator
    def join(self, inputs):
        """
        Testing whether attributes are excluded from collab to agg
        """
        # Aggregator attribute check
        validate = (
            hasattr(self, "include_agg_to_agg") is True
            and hasattr(self, "include_agg_to_collab") is True
            and hasattr(self, "exclude_agg_to_collab") is True
            and hasattr(self, "exclude_agg_to_agg") is False
        )

        # Collaborator attribute check
        for input in inputs:
            validation = validate and (
            hasattr(input, "include_collab_to_collab") is True
            and hasattr(input, "exclude_collab_to_collab") is False
            and hasattr(input, "exclude_collab_to_agg") is False
            and hasattr(input, "include_collab_to_agg") is True
            )
        assert validation, "Exclude test failed in join"
        log.info("Exclude test passed in join")
        self.next(self.end)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """
        log.info("Testing FederatedFlow - Ending Test for Exclude Attributes")
