# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator
import numpy as np


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


class TestFlowPrivateAttributes(FLSpec):
    """
    Testflow to validate Aggregator private attributes are not accessible to collaborators
    and vice versa
    """

    error_list = []

    @aggregator
    def start(self):
        """
        Flow start.
        """
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - Starting Test for accessibility of private "
            + f"attributes  {bcolors.ENDC}"
        )
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
        self.next(
            self.collaborator_step_b, exclude=["exclude_collab_to_collab"]
        )

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
        if hasattr(self, "test_loader") is False:
            TestFlowPrivateAttributes.error_list.append(
                "aggregator_join_aggregator_attributes_missing"
            )
            print(
                f"{bcolors.FAIL} ... Attribute test failed in join - aggregator private attributes"
                + f" not accessible {bcolors.ENDC}"
            )

        for input in enumerate(inputs):
            if (
                hasattr(input, "train_loader") is True
                or hasattr(input, "test_loader") is True
            ):
                # Error - we are able to access collaborator attributes
                TestFlowPrivateAttributes.error_list.append(
                    "join_collaborator_attributes_found"
                )
                print(
                    f"{bcolors.FAIL} ... Attribute test failed in Join - COllaborator: {collab}"
                    + f" private attributes accessible {bcolors.ENDC}"
                )

        self.next(self.end)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - Ending Test  for accessibility of private "
            + f"attributes  {bcolors.ENDC}"
        )

        if TestFlowPrivateAttributes.error_list:
            raise (
                AssertionError(
                    f"{bcolors.FAIL}\n ...Test case failed ... {bcolors.ENDC}"
                )
            )
        else:
            print(f"{bcolors.OKGREEN}\n ...Test case passed ... {bcolors.ENDC}")

        TestFlowPrivateAttributes.error_list = []


def validate_collab_private_attr(self, private_attr, step_name):
    # Aggregator should only be able to access its own attributes
    if hasattr(self, private_attr) is False:
        TestFlowPrivateAttributes.error_list.append(
            step_name + "_aggregator_attributes_missing"
        )
        print(
            f"{bcolors.FAIL} ...Failed in {step_name} - aggregator private attributes not "
            + f"accessible {bcolors.ENDC}"
        )

    for idx, collab in enumerate(self.collaborators):
        # Collaborator private attributes should not be accessible
        if (
            type(self.collaborators[idx]) is not str
            or hasattr(self.runtime, "_collaborators") is True
            or hasattr(self.runtime, "__collaborators") is True
        ):
            # Error - we are able to access collaborator attributes
            TestFlowPrivateAttributes.error_list.append(
                step_name + "_collaborator_attributes_found"
            )
            print(
                f"{bcolors.FAIL} ... Attribute test failed in {step_name} - collaborator {collab} "
                + f"private attributes accessible {bcolors.ENDC}"
            )


def validate_agg_private_attrs(self, private_attr_1, private_attr_2, step_name):
    # Collaborator should only be able to access its own attributes
    if (
        hasattr(self, private_attr_1) is False
        or hasattr(self, private_attr_2) is False
    ):
        TestFlowPrivateAttributes.error_list.append(
            step_name + "collab_attributes_not_found"
        )
        print(
            f"{bcolors.FAIL} ... Attribute test failed in {step_name} - Collab "
            + f"private attributes not accessible {bcolors.ENDC}"
        )

    if hasattr(self.runtime, "_aggregator") is True:
        # Error - we are able to access aggregator attributes
        TestFlowPrivateAttributes.error_list.append(
            step_name + "_aggregator_attributes_found"
        )
        print(
            f"{bcolors.FAIL} ... Attribute test failed in {step_name} - Aggregator"
            + f" private attributes accessible {bcolors.ENDC}"
        )


if __name__ == "__main__":
    # Setup Aggregator with private attributes
    aggregator = Aggregator()
    aggregator.private_attributes = {
        "test_loader": np.random.rand(10, 28, 28)  # Random data
    }

    # Setup collaborators with private attributes
    collaborator_names = [
        "Portland",
        "Seattle",
        "Chandler",
        "Bangalore",
        "Delhi",
        "Paris",
        "New York",
        "Tel Aviv",
        "Beijing",
        "Tokyo",
    ]
    collaborators = [Collaborator(name=name) for name in collaborator_names]
    for idx, collab in enumerate(collaborators):
        collab.private_attributes = {
            "train_loader": np.random.rand(idx * 50, 28, 28),
            "test_loader": np.random.rand(idx * 10, 28, 28),
        }

    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators
    )
    print(f"Local runtime collaborators = {local_runtime.collaborators}")

    flflow = TestFlowPrivateAttributes(checkpoint=False)
    flflow.runtime = local_runtime
    for i in range(5):
        print(f"Starting round {i}...")
        flflow.run()

    print(f"{bcolors.OKBLUE}End of Testing FederatedFlow {bcolors.ENDC}")
