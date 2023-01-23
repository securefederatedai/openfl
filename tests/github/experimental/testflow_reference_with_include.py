# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator

import sys
import torch.nn as nn
import torch.optim as optim
import inspect
from types import MethodType

MIN_COLLECTION_COUNT = 2


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(60, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TestFlowReferenceWithInclude(FLSpec):

    """
    Testflow to validate references of collabartor attributes in Federated Flow with include.

    """

    step_one_collab_attrs = {}
    step_two_collab_attrs = {}
    all_ref_error_dict = {}

    @aggregator
    def start(self):
        """
        Flow start.

        """
        self.agg_agg_attr_dict = {key: key for key in range(5)}
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - Starting Test for validating references "
            + f"{bcolors.ENDC}"
        )
        self.next(self.test_create_agg_attr, include=["agg_agg_attr_dict"])

    @aggregator
    def test_create_agg_attr(self):

        """
        Create different types of objects
        """

        self.agg_attr_list = [1, 2, 5, 6, 7, 8]
        self.agg_attr_dict = {key: key for key in range(5)}

        self.agg_attr_model = Net()
        self.agg_attr_optimizer = optim.SGD(
            self.agg_attr_model.parameters(), lr=1e-3, momentum=1e-2
        )
        self.collaborators = self.runtime.collaborators
        self.next(
            self.test_create_collab_attr,
            foreach="collaborators",
            include=["collaborators", "agg_attr_list"],
        )

    @collaborator
    def test_create_collab_attr(self):

        """
        Modify the attirbutes of aggregator to validate the references.
        Create different types of objects.
        """

        self.collab_attr_list_one = [1, 2, 5, 6, 7, 8]
        self.collab_attr_dict_one = {key: key for key in range(5)}

        # append self attributes of collaborators
        attr_collab_dict, collab_attr_list = create_collab_dict(self)
        TestFlowReferenceWithInclude.step_one_collab_attrs.update(
            attr_collab_dict
        )

        if (
            len(TestFlowReferenceWithInclude.step_one_collab_attrs)
            >= MIN_COLLECTION_COUNT
        ):
            matched_ref_dict = find_match_ref_at_step(
                collab_attr_list,
                TestFlowReferenceWithInclude.step_one_collab_attrs,
            )
            validate_references(matched_ref_dict)

        # must be tested with include functionality
        self.next(
            self.test_create_more_collab_attr, include=["collab_attr_dict_one"]
        )

    @collaborator
    def test_create_more_collab_attr(self):
        """
        Create different types of objects.
        """

        self.collab_attr_list_two = [1, 2, 3, 5, 6, 8]
        self.collab_attr_dict_two = {key: key for key in range(5)}

        attr_collab_dict, collab_attr_list = create_collab_dict(self)
        TestFlowReferenceWithInclude.step_two_collab_attrs.update(
            attr_collab_dict
        )

        if (
            len(TestFlowReferenceWithInclude.step_two_collab_attrs)
            >= MIN_COLLECTION_COUNT
        ):
            matched_ref_dict = find_match_ref_at_step(
                collab_attr_list,
                TestFlowReferenceWithInclude.step_two_collab_attrs,
            )
            validate_references(matched_ref_dict)

        self.next(self.join, include=["collab_attr_dict_two"])

    @aggregator
    def join(self, inputs):
        """
        Iterate over the references of collaborator attributes
        validate uniqueness of attributes and raise assertion
        """

        all_attr_list = filter_attrs(inspect.getmembers(inputs[0]))

        matched_ref_dict = find_matched_references(all_attr_list, inputs)
        validate_references(matched_ref_dict)
        all_shared_attr = ""
        print(f"\n{bcolors.UNDERLINE}Reference test summary: {bcolors.ENDC}\n")
        for key, val in TestFlowReferenceWithInclude.all_ref_error_dict.items():
            all_shared_attr = all_shared_attr + ",".join(val)
        if all_shared_attr:
            print(
                f"{bcolors.FAIL}...Test case failed for {all_shared_attr} {bcolors.ENDC}"
            )
        else:
            print(
                f"{bcolors.OKGREEN}...Test case passed for all the attributes."
            )
        self.next(self.end)

    @aggregator
    def end(self):
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - Ending test for validatng the references. "
            + f"{bcolors.ENDC}"
        )
        if TestFlowReferenceWithInclude.all_ref_error_dict:
            raise (
                AssertionError(
                    f"{bcolors.FAIL}\n ...Test case failed ... {bcolors.ENDC}"
                )
            )

        TestFlowReferenceWithInclude.step_one_collab_attrs = {}
        TestFlowReferenceWithInclude.step_two_collab_attrs = {}
        TestFlowReferenceWithInclude.all_ref_error_dict = {}


def filter_attrs(attr_list):
    valid_attrs = []
    reserved_words = ["next", "runtime", "execute_next"]
    for attr in attr_list:
        if (
            not attr[0].startswith("_")
            and attr[0] not in reserved_words
            and not hasattr(TestFlowReferenceWithInclude, attr[0])
        ):
            if not isinstance(attr[1], MethodType):
                valid_attrs.append(attr[0])
    return valid_attrs


def find_matched_references(collab_attr_list, all_collborators):
    """
    Iterate attributes of collborator and capture the duplicate reference
    """
    matched_ref_dict = {}
    previous_collaborator = ""
    # Initialize dictionary with collborator as key and value as empty list to hold
    # duplicated attr list
    for collborator_name in all_collborators:
        matched_ref_dict[collborator_name.input] = []

    # Iterate the attributes and get duplicate attribute id
    for attr in collab_attr_list:
        attr_dict = {attr: []}
        for collab in all_collborators:
            attr_id = id(getattr(collab, attr))
            collaborator_name = collab.input
            if attr_id not in attr_dict.get(attr):
                attr_dict.get(attr).append(attr_id)
            else:
                # append the dict with collabartor as key and attrs as value having same reference
                matched_ref_dict.get(collaborator_name).append(attr)
                print(
                    f"{bcolors.FAIL} ... Reference test failed - {collaborator_name} sharing same "
                    + f"{attr} reference with {previous_collaborator} {bcolors.ENDC}"
                )
            previous_collaborator = collaborator_name
    return matched_ref_dict


def validate_references(matched_ref_dict):
    """
    Iterate reference list and raise assertion for conflicts
    """
    collborators_sharing_ref = []
    reference_flag = False

    for collab, val in matched_ref_dict.items():
        if val:
            collborators_sharing_ref.append(collab)
            reference_flag = True
    if collborators_sharing_ref:
        for collab in collborators_sharing_ref:
            if collab not in TestFlowReferenceWithInclude.all_ref_error_dict:
                TestFlowReferenceWithInclude.all_ref_error_dict[
                    collab
                ] = matched_ref_dict.get(collab)

    if not reference_flag:
        print(
            f"{bcolors.OKGREEN}  Pass : Reference test passed  {bcolors.ENDC}"
        )


def create_collab_dict(collab):
    """
    saving the collaborator and its attributes to compare with other collaborator refences.
    return : dict ({
                    'Portland': {'collab_attr_dict_one': 140512653871680},
                    'Seattle': {'collab_attr_dict_one': 140512653871936}
                })
    """
    attr_collab_dict = {}
    collab_attr_list = filter_attrs(inspect.getmembers(collab))
    for attr in collab_attr_list:
        attr_id = id(getattr(collab, attr))
        if attr_collab_dict.get(collab.input):
            attr_collab_dict.get(collab.input)[attr] = attr_id
        else:
            attr_collab_dict[collab.input] = {}
            attr_collab_dict.get(collab.input)[attr] = attr_id
    return attr_collab_dict, collab_attr_list


def find_match_ref_at_step(collab_attr_list, all_collborators):
    collab_names = all_collborators.keys()

    matched_ref_dict = {}
    for collborator_name in collab_names:
        matched_ref_dict[collborator_name] = []

    previous_collaborator = ""
    for attr in collab_attr_list:
        attr_dict = {attr: []}
        for collborator_name in all_collborators.keys():
            attr_id = all_collborators[collborator_name][attr]
            if attr_id not in attr_dict.get(attr):
                attr_dict.get(attr).append(attr_id)
            else:
                matched_ref_dict.get(collborator_name).append(attr)
                print(
                    f"{bcolors.FAIL} ... Reference test failed - {collborator_name} sharing same "
                    + f"{attr} reference with {previous_collaborator} {bcolors.ENDC}"
                )

            previous_collaborator = collborator_name

    return matched_ref_dict


if __name__ == "__main__":

    # Setup participants
    aggregator = Aggregator()
    aggregator.private_attributes = {}

    # Setup collaborators with private attributes
    collaborator_names = ["Portland", "Seattle", "Chandler", "Bangalore"]
    collaborators = [Collaborator(name=name) for name in collaborator_names]
    collaborator.private_attributes = {}

    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators
    )

    if len(sys.argv) > 1:
        if sys.argv[1] == 'ray':
            local_runtime = LocalRuntime(
                aggregator=aggregator, collaborators=collaborators, backend='ray'
            )

    print(f"Local runtime collaborators = {local_runtime.collaborators}")

    testflow = TestFlowReferenceWithInclude(checkpoint=False)
    testflow.runtime = local_runtime

    for i in range(5):
        print(f"Starting round {i}...")
        testflow.run()
