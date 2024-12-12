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
        if (
            hasattr(self, "include_agg_to_agg") is True
            and hasattr(self, "exclude_agg_to_agg") is False
        ):
            log.info("... Exclude test passed in test_exclude_agg_to_agg")
        else:
            log.error("... Exclude test failed in test_exclude_agg_to_agg")
            raise ValueError("test_exclude_agg_to_agg")

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

        if (
            hasattr(self, "include_agg_to_agg") is True
            and hasattr(self, "include_agg_to_collab") is True
            and hasattr(self, "exclude_agg_to_agg") is False
            and hasattr(self, "exclude_agg_to_collab") is False
        ):
            log.info("... Exclude test passed in test_exclude_agg_to_collab")
        else:
            log.error("... Exclude test failed in test_exclude_agg_to_collab")
            raise ValueError("test_exclude_agg_to_collab")

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

        if (
            hasattr(self, "include_agg_to_agg") is True
            and hasattr(self, "include_agg_to_collab") is True
            and hasattr(self, "include_collab_to_collab") is True
            and hasattr(self, "exclude_agg_to_agg") is False
            and hasattr(self, "exclude_agg_to_collab") is False
            and hasattr(self, "exclude_collab_to_collab") is False
        ):
            log.info("... Exclude test passed in test_exclude_collab_to_collab")
        else:
            log.error("... Exclude test failed in test_exclude_collab_to_collab")
            raise ValueError("test_exclude_collab_to_collab")

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

        if validation:
            log.info("... Exclude test passed in join")
        else:
            log.error("... Exclude test failed in join")
            raise ValueError("join")

        log.info("Exclude attribute test summary:")
        self.next(self.end)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """
        log.info("Testing FederatedFlow - Ending Test for Exclude Attributes")
