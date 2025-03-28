# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
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


class TestFlowExclude(FLSpec):
    """
    Testflow to validate exclude functionality in Federated Flow
    """

    exclude_error_list = []

    @aggregator
    def start(self):
        """
        Flow start.
        """
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - Starting Test for Exclude Attributes "
            + f"{bcolors.ENDC}"
        )
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
            print(
                f"{bcolors.OKGREEN} ... Exclude test passed in test_exclude_agg_to_agg "
                + f"{bcolors.ENDC}"
            )
        else:
            TestFlowExclude.exclude_error_list.append("test_exclude_agg_to_agg")
            print(
                f"{bcolors.FAIL} ... Exclude test failed in test_exclude_agg_to_agg {bcolors.ENDC}"
            )

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
            print(
                f"{bcolors.OKGREEN} ... Exclude test passed in test_exclude_agg_to_collab "
                + f"{bcolors.ENDC}"
            )
        else:
            TestFlowExclude.exclude_error_list.append("test_exclude_agg_to_collab")
            print(
                f"{bcolors.FAIL} ... Exclude test failed in test_exclude_agg_to_collab "
                + f"{bcolors.ENDC}"
            )

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
            print(
                f"{bcolors.OKGREEN} ... Exclude test passed in test_exclude_collab_to_collab "
                + f"{bcolors.ENDC}"
            )
        else:
            TestFlowExclude.exclude_error_list.append("test_exclude_collab_to_collab")
            print(
                f"{bcolors.FAIL} ... Exclude test failed in test_exclude_collab_to_collab "
                + f"{bcolors.ENDC}"
            )

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
            print(f"{bcolors.OKGREEN} ... Exclude test passed in join {bcolors.ENDC}")
        else:
            TestFlowExclude.exclude_error_list.append("join")
            print(f"{bcolors.FAIL} ... Exclude test failed in join {bcolors.ENDC}")

        print(f"\n{bcolors.UNDERLINE}Exclude attribute test summary: {bcolors.ENDC}\n")

        if TestFlowExclude.exclude_error_list:
            validated_exclude_variables = ", ".join(TestFlowExclude.exclude_error_list)
            print(
                f"{bcolors.FAIL}...Test case failed for {validated_exclude_variables} "
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
            f"{bcolors.OKBLUE}Testing FederatedFlow - Ending Test for Exclude Attributes "
            + f"{bcolors.ENDC}"
        )
        if TestFlowExclude.exclude_error_list:
            raise (
                AssertionError(
                    f"{bcolors.FAIL}\n ...Test case failed ... {bcolors.ENDC}"
                )
            )


if __name__ == "__main__":
    # Setup participants
    aggregator = Aggregator()

    # Setup collaborators
    collaborator_names = ["Portland", "Chandler", "Bangalore", "Delhi"]
    collaborators = []
    for collaborator_name in collaborator_names:
        collaborators.append(Collaborator(name=collaborator_name))

    local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators)

    if len(sys.argv) > 1:
        if sys.argv[1] == "ray":
            local_runtime = LocalRuntime(
                aggregator=aggregator, collaborators=collaborators, backend="ray"
            )

    print(f"Local runtime collaborators = {local_runtime.collaborators}")

    flflow = TestFlowExclude(checkpoint=True)
    flflow.runtime = local_runtime
    for i in range(5):
        print(f"Starting round {i}...")
        flflow.run()

    print(f"{bcolors.OKBLUE}End of Testing FederatedFlow {bcolors.ENDC}")
