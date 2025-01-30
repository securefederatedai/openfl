# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TestFlowInclude(FLSpec):
    """
    Testflow to validate include functionality in Federated Flow
    """
    __test__ = False # to prevent pytest from trying to discover tests in the class

    @aggregator
    def start(self):
        """
        Flow start.
        """
        log.info("Testing FederatedFlow - Starting Test for Include Attributes")
        self.collaborators = self.runtime.collaborators

        self.exclude_agg_to_agg = 10
        self.include_agg_to_agg = 100
        self.next(
            self.test_include_agg_to_agg,
            include=["include_agg_to_agg", "collaborators"],
        )

    @aggregator
    def test_include_agg_to_agg(self):
        """
        Testing whether attributes are included from agg to agg
        """
        assert hasattr(self, "include_agg_to_agg") and not hasattr(self, "exclude_agg_to_agg"), \
            "Include test failed in test_include_agg_to_agg"

        log.info("Include test passed in test_include_agg_to_agg")

        self.include_agg_to_collab = 100
        self.exclude_agg_to_collab = 78
        self.next(
            self.test_include_agg_to_collab,
            foreach="collaborators",
            include=["include_agg_to_collab", "collaborators"],
        )

    @collaborator
    def test_include_agg_to_collab(self):
        """
        Testing whether attributes are included from agg to collab
        """
        assert not hasattr(self, "include_agg_to_agg") and not hasattr(self, "exclude_agg_to_agg") \
               and not hasattr(self, "exclude_agg_to_collab") and hasattr(self, "include_agg_to_collab"), \
            "Include test failed in test_include_agg_to_collab"

        log.info("Include test passed in test_include_agg_to_collab")

        self.exclude_collab_to_collab = 10
        self.include_collab_to_collab = 44
        self.next(
            self.test_include_collab_to_collab,
            include=["include_collab_to_collab"],
        )

    @collaborator
    def test_include_collab_to_collab(self):
        """
        Testing whether attributes are included from collab to collab
        """
        assert not hasattr(self, "include_agg_to_agg") and not hasattr(self, "include_agg_to_collab") \
               and hasattr(self, "include_collab_to_collab") and not hasattr(self, "exclude_agg_to_agg") \
               and not hasattr(self, "exclude_agg_to_collab") and not hasattr(self, "exclude_collab_to_collab"), \
            "Include test failed in test_include_collab_to_collab"

        log.info("Include test passed in test_include_collab_to_collab")

        self.exclude_collab_to_agg = 20
        self.include_collab_to_agg = 56
        self.next(self.join, include=["include_collab_to_agg"])

    @aggregator
    def join(self, inputs):
        """
        Testing whether attributes are included from collab to agg
        """
        validate = hasattr(self, "include_agg_to_agg") and hasattr(self, "include_agg_to_collab") \
                   and hasattr(self, "exclude_agg_to_collab") and not hasattr(self, "exclude_agg_to_agg")

        for input in inputs:
            validation = validate and not hasattr(input, "include_collab_to_collab") \
                         and not hasattr(input, "exclude_collab_to_collab") \
                         and not hasattr(input, "exclude_collab_to_agg") \
                         and hasattr(input, "include_collab_to_agg")

        assert validation, "Include test failed in join"
        log.info("Include test passed in join")
        self.next(self.end)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """
        log.info("Testing FederatedFlow - Ending Test for Include Attributes")
