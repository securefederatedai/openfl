# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator


class bcolors:  # NOQA: N801
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class TestFlowIncludeExclude(FLSpec):
    """
    Testflow to validate include and exclude functionality in Federated Flow.
    """

    include_exclude_error_list = []

    def __init__(self, checkpoint: bool = False):
        super().__init__(checkpoint)

    @aggregator
    def start(self):
        """
        Flow start.
        """
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - Starting Test for Include and Exclude "
            + f"Attributes {bcolors.ENDC}"
        )
        self.collaborators = self.runtime.collaborators

        self.exclude_agg_to_agg = 10
        self.include_agg_to_agg = 100
        self.next(self.test_include_exclude_agg_to_agg, exclude=["exclude_agg_to_agg"])

    @aggregator
    def test_include_exclude_agg_to_agg(self):
        """
        Testing whether attributes are excluded from agg to agg
        """
        if (
            hasattr(self, "include_agg_to_agg") is True
            and hasattr(self, "exclude_agg_to_agg") is False
        ):
            print(
                f"{bcolors.OKGREEN} ... Exclude test passed in test_include_exclude_agg_to_agg "
                + f"{bcolors.ENDC}"
            )
        else:
            TestFlowIncludeExclude.include_exclude_error_list.append(
                "test_include_exclude_agg_to_agg"
            )
            print(
                f"{bcolors.FAIL} ... Exclude test failed in test_incude_exclude_agg_to_agg "
                + f"{bcolors.ENDC}"
            )

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
        if (
            hasattr(self, "include_agg_to_agg") is False
            and hasattr(self, "exclude_agg_to_agg") is False
            and hasattr(self, "exclude_agg_to_collab") is False
            and hasattr(self, "include_agg_to_collab") is True
        ):
            print(
                f"{bcolors.OKGREEN} ... Include test passed in test_include_exclude_agg_to_collab "
                + f"{bcolors.ENDC}"
            )
        else:
            TestFlowIncludeExclude.include_exclude_error_list.append(
                "test_incude_exclude_agg_to_collab"
            )
            print(
                f"{bcolors.FAIL} ... Include test failed in test_include_exclude_agg_to_collab "
                + f"{bcolors.ENDC}"
            )
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
        if (
            hasattr(self, "include_agg_to_agg") is False
            and hasattr(self, "include_agg_to_collab") is True
            and hasattr(self, "include_collab_to_collab") is True
            and hasattr(self, "exclude_agg_to_agg") is False
            and hasattr(self, "exclude_agg_to_collab") is False
            and hasattr(self, "exclude_collab_to_collab") is False
        ):
            print(
                f"{bcolors.OKGREEN} ... Exclude test passed in "
                + f"test_include_exclude_collab_to_collab {bcolors.ENDC}"
            )
        else:
            TestFlowIncludeExclude.include_exclude_error_list.append(
                "test_incude_exclude_collab_to_collab"
            )
            print(
                f"{bcolors.FAIL} ... Exclude test failed in test_include_exclude_collab_to_collab "
                + f"{bcolors.ENDC}"
            )

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

        if validation:
            print(
                f"{bcolors.OKGREEN} ... Include and Exclude tests passed in join {bcolors.ENDC}"
            )
        else:
            TestFlowIncludeExclude.include_exclude_error_list.append("join")
            print(
                f"{bcolors.FAIL} ... Include and Exclude tests failed in join {bcolors.ENDC}"
            )

        print(
            f"\n{bcolors.UNDERLINE} Include and exclude attributes test summary: {bcolors.ENDC}\n"
        )

        if TestFlowIncludeExclude.include_exclude_error_list:
            validated_include_exclude_variables = ",".join(
                TestFlowIncludeExclude.include_exclude_error_list
            )
            print(
                f"{bcolors.FAIL} ...Test case failed for {validated_include_exclude_variables} "
                + f"{bcolors.ENDC}"
            )

        self.next(self.end)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - Ending Test for Include and Exclude "
            + f"Attributes {bcolors.ENDC}"
        )
        if TestFlowIncludeExclude.include_exclude_error_list:
            raise (
                AssertionError(
                    f"{bcolors.FAIL}\n ...Test case failed ... {bcolors.ENDC}"
                )
            )
        print(f"{bcolors.OKBLUE}End of Testing FederatedFlow {bcolors.ENDC}")
