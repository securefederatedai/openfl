# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator

log = logging.getLogger(__name__)

class TestFlowIncludeExclude(FLSpec):
    """
    Testflow to validate include and exclude functionality in Federated Flow.
    """

    @aggregator
    def start(self):
        """
        Flow start.
        """
        log.info("Testing FederatedFlow - Starting Test for Include and Exclude Attributes")
        self.collaborators = self.runtime.collaborators

        self.exclude_agg_to_agg = 10
        self.include_agg_to_agg = 100
        self.next(self.test_include_exclude_agg_to_agg, exclude=["exclude_agg_to_agg"])

    @aggregator
    def test_include_exclude_agg_to_agg(self):
        """
        Testing whether attributes are excluded from agg to agg
        """
        assert hasattr(self, "include_agg_to_agg") is True and hasattr(self, "exclude_agg_to_agg") is False, \
            "Exclude test failed in test_include_exclude_agg_to_agg"
        log.info("Exclude test passed in test_include_exclude_agg_to_agg")

        self.include_agg_to_collab = 100
        self.exclude_agg_to_collab = 78
        self.next(
            self.test_include_exclude_agg_to_collab,
            foreach="collaborators",
            include=["include_agg_to_collab", "collaborators"],
        )

    @collaborator
    def test_include_exclude_agg_to_collab(self):
        """
        Testing whether attributes are included from agg to collab
        """
        assert (
            hasattr(self, "include_agg_to_agg") is False
            and hasattr(self, "exclude_agg_to_agg") is False
            and hasattr(self, "exclude_agg_to_collab") is False
            and hasattr(self, "include_agg_to_collab") is True
        ), "Include test failed in test_include_exclude_agg_to_collab"
        log.info("Include test passed in test_include_exclude_agg_to_collab")

        self.exclude_collab_to_collab = 10
        self.include_collab_to_collab = 44
        self.next(
            self.test_include_exclude_collab_to_collab,
            exclude=["exclude_collab_to_collab"],
        )

    @collaborator
    def test_include_exclude_collab_to_collab(self):
        """
        Testing whether attributes are excluded from collab to collab
        """
        assert (
            hasattr(self, "include_agg_to_agg") is False
            and hasattr(self, "include_agg_to_collab") is True
            and hasattr(self, "include_collab_to_collab") is True
            and hasattr(self, "exclude_agg_to_agg") is False
            and hasattr(self, "exclude_agg_to_collab") is False
            and hasattr(self, "exclude_collab_to_collab") is False
        ), "Exclude test failed in test_include_exclude_collab_to_collab"
        log.info("Exclude test passed in test_include_exclude_collab_to_collab")

        self.exclude_collab_to_agg = 20
        self.include_collab_to_agg = 56
        self.next(self.join, include=["include_collab_to_agg"])

    @aggregator
    def join(self, inputs):
        """
        Testing whether attributes are included from collab to agg
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
                hasattr(input, "include_collab_to_collab") is False
                and hasattr(input, "exclude_collab_to_collab") is False
                and hasattr(input, "exclude_collab_to_agg") is False
                and hasattr(input, "include_collab_to_agg") is True
            )

        assert validation, "Include and Exclude tests failed in join"
        log.info("Include and Exclude tests passed in join")
        self.next(self.end)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """
        log.info("Testing FederatedFlow - Ending Test for Include and Exclude Attributes")
