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


class TestFlowPrivateAttributes(FLSpec):
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

        validate_agg_private_attr(self, "start", aggr=["test_loader_agg_via_callable"], collabs=["train_loader_via_callable", "test_loader_via_callable"])

        self.exclude_agg_to_agg = 10
        self.include_agg_to_agg = 100
        self.next(self.aggregator_step, exclude=["exclude_agg_to_agg"])

    @aggregator
    def aggregator_step(self):
        """
        Testing whether Agg private attributes are accessible in next agg step.
        Collab private attributes should not be accessible here
        """
        validate_agg_private_attr(self, "aggregator_step", aggr=["test_loader_agg_via_callable"], collabs=["train_loader_via_callable", "test_loader_via_callable"])

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
        validate_collab_private_attrs(
            self, "collaborator_step_a", aggr=["test_loader_agg_via_callable"], collabs=["train_loader_via_callable", "test_loader_via_callable"]
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

        validate_collab_private_attrs(
            self, "collaborator_step_b", aggr=["test_loader_agg_via_callable"], collabs=["train_loader_via_callable", "test_loader_via_callable"]
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
        if hasattr(self, "test_loader_agg_via_callable") is False:
            TestFlowPrivateAttributes.ERROR_LIST.append(
                "aggregator_join_aggregator_attributes_missing"
            )
            print(
                f"{bcolors.FAIL} ... Attribute test failed in join - aggregator private attributes"
                + f" not accessible {bcolors.ENDC}"
            )

        for input in enumerate(inputs):
            collab = input[1].input
            if (
                hasattr(input, "train_loader_via_callable")
                or hasattr(input, "test_loader_via_callable")
            ):
                # Error - we are able to access collaborator attributes
                TestFlowPrivateAttributes.ERROR_LIST.append(
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

        if TestFlowPrivateAttributes.ERROR_LIST:
            raise (
                AssertionError(
                    f"{bcolors.FAIL}\n ...Test case failed ... {bcolors.ENDC}"
                )
            )
        else:
            print(f"{bcolors.OKGREEN}\n ...Test case passed ... {bcolors.ENDC}")

        TestFlowPrivateAttributes.ERROR_LIST = []


def validate_agg_private_attr(self, step_name, **private_attrs_kwargs):

    """
        Validate that aggregator can only access their own attributes

        Args:
            step_name: Name of the step being validated
            private_attr_kwargs: Keyword arguments specifying the names of private attributes for the aggregator and collaborators.
    """
    agg_attrs = private_attrs_kwargs.get('aggr', [])
    collab_attrs = private_attrs_kwargs.get('collabs', [])
    # Aggregator should only be able to access its own attributes

    # check for missing aggregator attributes
    inaccessible_agg_attrs = [attr for attr in agg_attrs if not hasattr(self, attr)]
    if inaccessible_agg_attrs:
        TestFlowPrivateAttributes.ERROR_LIST.append(
            step_name + " aggregator_attributes_missing"
        )
        print(
            f"{bcolors.FAIL} ...Failed in {step_name} - aggregator private attributes not "
            + f"accessible {bcolors.ENDC}"
        )

    # check for collaborator private attributes that should not be accessible
    breached_collab_attrs = [attr for attr in collab_attrs if hasattr(self, attr)]
    if breached_collab_attrs:
        TestFlowPrivateAttributes.ERROR_LIST.append(
            step_name + "_collaborator_attributes_found"
        )
        print(
            f"{bcolors.FAIL} ... Attribute test failed in {step_name} - collaborator"
            + f"private attributes accessible:{','.join(breached_collab_attrs)} {bcolors.ENDC}"
        )
    for idx, collab in enumerate(self.collaborators):
        # Collaborator attributes should not be accessible in aggregator step
        if (
            type(self.collaborators[idx]) is not str
            or hasattr(self.runtime, "_collaborators")
            or hasattr(self.runtime, "__collaborators")
        ):
            # Error - we are able to access collaborator attributes
            TestFlowPrivateAttributes.ERROR_LIST.append(
                step_name + "_collaborator_attributes_found"
            )
            print(
                f"{bcolors.FAIL} ... Attribute test failed in {step_name} - collaborator {collab} "
                + f"private attributes accessible {bcolors.ENDC}"
            )


def validate_collab_private_attrs(self, step_name, **private_attrs_kwargs):

    """
        Validate that collaborators can only access their own attributes

        Args:
            step_name: Name of the step being validated
            private_attr_kwargs: Keyword arguments specifying the names of private attributes for the aggregator and collaborators.
    """
    agg_attrs = private_attrs_kwargs.get('aggr', [])
    collab_attrs = private_attrs_kwargs.get('collabs', [])

    # Collaborator should only be able to access its own attributes

    # check for missing collaborators attributes
    inaccessible_collab_attrs = [attr for attr in collab_attrs if not hasattr(self, attr)]

    if inaccessible_collab_attrs:
        TestFlowPrivateAttributes.ERROR_LIST.append(
            step_name + " collab_attributes_not_found"
        )
        print(
            f"{bcolors.FAIL} ... Attribute test failed in {step_name} - Collab "
            + f"private attributes not accessible {bcolors.ENDC}"
        )
    # check for aggregator private attributes that should not be accessible
    breached_agg_attr = [attr for attr in agg_attrs if hasattr(self, attr)]
    if breached_agg_attr:
        TestFlowPrivateAttributes.ERROR_LIST.append(
            step_name + "_aggregator_attributes_found"
        )

        print(
            f"{bcolors.FAIL} ... Attribute test failed in {step_name} - Aggregator"
            + f" private attributes accessible: {','.join(breached_agg_attr)} {bcolors.ENDC}"
        )

    # Aggregator attributes should not be accessible in collaborator step
    if hasattr(self.runtime, "_aggregator") and isinstance(self.runtime._aggregator, Aggregator):
        # Error - we are able to access aggregator attributes
        TestFlowPrivateAttributes.ERROR_LIST.append(
            step_name + "_aggregator_attributes_found"
        )
        print(
            f"{bcolors.FAIL} ... Attribute test failed in {step_name} - Aggregator"
            + f" private attributes accessible {bcolors.ENDC}"
        )
