# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.workflow.component import Aggregator
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


class TestFlowParticipantName(FLSpec):
    """
    Testflow to validate Aggregator private attributes are not accessible to collaborators
    and vice versa
    """

    ERROR_LIST = []

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
        if hasattr(self, "test_loader") is False:
            TestFlowParticipantName.ERROR_LIST.append(
                "aggregator_join_aggregator_attributes_missing"
            )
            print(
                f"{bcolors.FAIL} ... Attribute test failed in join - aggregator private attributes"
                + f" not accessible {bcolors.ENDC}"
            )

        for input in enumerate(inputs):
            collab = input[1].input
            if (
                hasattr(input, "train_loader") is True
                or hasattr(input, "test_loader") is True
            ):
                # Error - we are able to access collaborator attributes
                TestFlowParticipantName.ERROR_LIST.append(
                    "join_collaborator_attributes_found"
                )
                print(
                    f"{bcolors.FAIL} ... Attribute test failed in Join - Collaborator: {collab}"
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

        if TestFlowParticipantName.ERROR_LIST:
            raise (
                AssertionError(
                    f"{bcolors.FAIL}\n ...Test case failed ... {bcolors.ENDC}"
                )
            )
        else:
            print(f"{bcolors.OKGREEN}\n ...Test case passed ... {bcolors.ENDC}")

        TestFlowParticipantName.ERROR_LIST = []


def validate_collab_private_attr(self, private_attr, step_name):
    # Aggregator should only be able to access its own attributes
    if hasattr(self, private_attr) is False:
        TestFlowParticipantName.ERROR_LIST.append(
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
            TestFlowParticipantName.ERROR_LIST.append(
                step_name + "_collaborator_attributes_found"
            )
            print(
                f"{bcolors.FAIL} ... Attribute test failed in {step_name} - collaborator {collab} "
                + f"private attributes accessible {bcolors.ENDC}"
            )


def validate_agg_private_attrs(self, private_attr_1, private_attr_2, step_name):
    # Collaborator should only be able to access its own attributes
    if not hasattr(self, private_attr_1) or not hasattr(self, private_attr_2):
        TestFlowParticipantName.ERROR_LIST.append(
            step_name + "collab_attributes_not_found"
        )
        print(
            f"{bcolors.FAIL} ... Attribute test failed in {step_name} - Collab "
            + f"private attributes not accessible {bcolors.ENDC}"
        )

    if hasattr(self.runtime, "_aggregator") and isinstance(self.runtime._aggregator, Aggregator):
        # Error - we are able to access aggregator attributes
        TestFlowParticipantName.ERROR_LIST.append(
            step_name + "_aggregator_attributes_found"
        )
        print(
            f"{bcolors.FAIL} ... Attribute test failed in {step_name} - Aggregator"
            + f" private attributes accessible {bcolors.ENDC}"
        )
