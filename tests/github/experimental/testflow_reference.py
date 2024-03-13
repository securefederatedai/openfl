# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator

import sys
import io
import math
import logging
import torch.nn as nn
import torch.optim as optim
import inspect
from types import MethodType


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


class TestFlowReference(FLSpec):

    """
    Testflow to validate references of collabartor attributes in Federated Flow.

    """

    step_one_collab_attrs = []
    step_two_collab_attrs = []
    all_ref_error_dict = {}
    agg_attr_dict = {}

    @aggregator
    def start(self):
        """
        Flow start.

        """
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - Starting Test for validating references. "
            + f"{bcolors.ENDC}"
        )
        self.next(self.test_create_agg_attr)

    @aggregator
    def test_create_agg_attr(self):
        """
        Create different types of objects.
        """

        self.agg_attr_int = 10
        self.agg_attr_str = "Test string data"
        self.agg_attr_list = [1, 2, 5, 6, 7, 8]
        self.agg_attr_dict = {key: key for key in range(5)}
        self.agg_attr_file = io.StringIO("Test file data in aggregator")
        self.agg_attr_math = math.sqrt(2)
        self.agg_attr_complex_num = complex(2, 3)
        self.agg_attr_log = logging.getLogger("Test logger data in aggregator")
        self.agg_attr_model = Net()
        self.agg_attr_optimizer = optim.SGD(
            self.agg_attr_model.parameters(), lr=1e-3, momentum=1e-2
        )
        self.collaborators = self.runtime.collaborators

        # get aggregator attributes
        agg_attr_list = filter_attrs(inspect.getmembers(self))
        for attr in agg_attr_list:
            agg_attr_id = id(getattr(self, attr))
            TestFlowReference.agg_attr_dict[attr] = agg_attr_id
        self.next(self.test_create_collab_attr, foreach="collaborators")

    @collaborator
    def test_create_collab_attr(self):
        """
        Modify the attirbutes of aggregator to validate the references.
        Create different types of objects.
        """

        self.agg_attr_int += self.index
        self.agg_attr_str = self.agg_attr_str + " " + self.input
        self.agg_attr_complex_num += complex(self.index, self.index)
        self.agg_attr_math += self.index
        self.agg_attr_log = " " + self.input

        self.collab_attr_int_one = 20 + self.index
        self.collab_attr_str_one = "Test string data in collab " + self.input
        self.collab_attr_list_one = [1, 2, 5, 6, 7, 8]
        self.collab_attr_dict_one = {key: key for key in range(5)}
        self.collab_attr_file_one = io.StringIO("Test file data in collaborator")
        self.collab_attr_math_one = math.sqrt(self.index)
        self.collab_attr_complex_num_one = complex(self.index, self.index)
        self.collab_attr_log_one = logging.getLogger(
            "Test logger data in collaborator " + self.input
        )

        # append attributes of collaborator
        TestFlowReference.step_one_collab_attrs.append(self)

        if len(TestFlowReference.step_one_collab_attrs) >= 2:
            collab_attr_list = filter_attrs(inspect.getmembers(self))
            matched_ref_dict = find_matched_references(
                collab_attr_list, TestFlowReference.step_one_collab_attrs
            )
            validate_collab_references(matched_ref_dict)

        self.next(self.test_create_more_collab_attr)

    @collaborator
    def test_create_more_collab_attr(self):
        """
        Create different types of objects.
        """

        self.collab_attr_int_two = 30 + self.index
        self.collab_attr_str_two = "String reference three " + self.input
        self.collab_attr_list_two = [1, 2, 3, 5, 6, 8]
        self.collab_attr_dict_two = {key: key for key in range(5)}
        self.collab_attr_file_two = io.StringIO("Test file reference one")
        self.collab_attr_math_two = math.sqrt(2)
        self.collab_attr_complex_num_two = complex(2, 3)
        self.collab_attr_log_two = logging.getLogger(
            "Test logger data in collaborator" + self.input
        )

        TestFlowReference.step_two_collab_attrs.append(self)

        if len(TestFlowReference.step_two_collab_attrs) >= 2:
            collab_attr_list = filter_attrs(inspect.getmembers(self))
            matched_ref_dict = find_matched_references(
                collab_attr_list, TestFlowReference.step_two_collab_attrs
            )
            validate_collab_references(matched_ref_dict)

        self.next(self.join)

    @aggregator
    def join(self, inputs):
        """
        Iterate over the references of collaborator attributes
        validate uniqueness of attributes and raise assertion
        """

        all_attr_list = filter_attrs(inspect.getmembers(inputs[0]))
        agg_attrs = filter_attrs(inspect.getmembers(self))

        # validate aggregator references are intact after coming out of collaborators.
        validate_agg_attr_ref(agg_attrs, self)

        # validate collaborators references are not shared in between.
        matched_ref_dict = find_matched_references(all_attr_list, inputs)
        validate_collab_references(matched_ref_dict)

        # validate aggregator references are not shared with any of the collaborators .
        validate_agg_collab_references(inputs, self, agg_attrs)

        all_shared_attr = ""
        print(f"\n{bcolors.UNDERLINE}Reference test summary: {bcolors.ENDC}\n")
        for val in TestFlowReference.all_ref_error_dict.values():
            all_shared_attr = all_shared_attr + ",".join(val)
        if all_shared_attr:
            print(
                f"{bcolors.FAIL}...Test case failed for {all_shared_attr} {bcolors.ENDC}"
            )
        else:
            print(
                f"{bcolors.OKGREEN}...Test case passed for all the attributes.{bcolors.ENDC}"
            )

        self.next(self.end)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - Ending test for validatng the references. "
            + f"{bcolors.ENDC}"
        )
        if TestFlowReference.all_ref_error_dict:
            raise (
                AssertionError(
                    f"{bcolors.FAIL}\n ...Test case failed ... {bcolors.ENDC}"
                )
            )

        TestFlowReference.step_one_collab_attrs = []
        TestFlowReference.step_two_collab_attrs = []
        TestFlowReference.all_ref_error_dict = {}


def filter_attrs(attr_list):
    valid_attrs = []
    reserved_words = ["next", "runtime", "execute_next"]
    for attr in attr_list:
        if (
            not attr[0].startswith("_")
            and attr[0] not in reserved_words
            and not hasattr(TestFlowReference, attr[0])
        ):
            if not isinstance(attr[1], MethodType):
                valid_attrs.append(attr[0])
    return valid_attrs


def find_matched_references(collab_attr_list, all_collaborators):
    """
    Iterate attributes of collborator and capture the duplicate reference
    return: dict: {
                    'Portland': ['failed attributes'], 'Seattle': [],
                  }
    """
    matched_ref_dict = {}
    for i in range(len(all_collaborators)):
        matched_ref_dict[all_collaborators[i].input] = []

    # For each attribute in the collaborator attribute list, check if any of the collaborator
    # attributes are shared with another collaborator
    for attr_name in collab_attr_list:
        for i, curr_collab in enumerate(all_collaborators):
            # Compare the current collaborator with the collaborator(s) that come(s) after it.
            for next_collab in all_collaborators[i + 1:]:
                # Check if both collaborators have the current attribute
                if hasattr(curr_collab, attr_name) and hasattr(next_collab, attr_name):
                    # Check if both collaborators are sharing same reference
                    if getattr(curr_collab, attr_name) is getattr(
                        next_collab, attr_name
                    ):
                        matched_ref_dict[curr_collab.input].append(attr_name)
                        print(
                            f"{bcolors.FAIL} ... Reference test failed - {curr_collab.input} \
                                sharing same "
                            + f"{attr_name} reference with {next_collab.input} {bcolors.ENDC}"
                        )

    return matched_ref_dict


def validate_collab_references(matched_ref_dict):
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
            if collab not in TestFlowReference.all_ref_error_dict:
                TestFlowReference.all_ref_error_dict[collab] = matched_ref_dict.get(
                    collab
                )

    if not reference_flag:
        print(
            f"{bcolors.OKGREEN}  Pass : Reference test passed for collaborators. {bcolors.ENDC}"
        )


def validate_agg_attr_ref(agg_attrs, agg_obj):
    """
    Verifies aggregator attributes are retained after
    collaborator execution
    """
    attr_flag = False
    for attr in agg_attrs:
        if TestFlowReference.agg_attr_dict.get(attr) == id(getattr(agg_obj, attr)):
            attr_flag = True
    if not attr_flag:
        print(
            f"{bcolors.FAIL}...Aggregator references are not intact after coming out of "
            + f"collaborators.{bcolors.ENDC}"
        )
    else:
        print(
            f"{bcolors.OKGREEN}  Pass : Aggregator references are intact after coming out of "
            + f"collaborators.{bcolors.ENDC}"
        )


def validate_agg_collab_references(all_collborators, agg_obj, agg_attrs):
    """
    Iterate attributes of aggregator and collborator to capture the mismatched references.
    """

    mis_matched_ref = {}
    for collab in all_collborators:
        mis_matched_ref[collab.input] = []

    attr_ref_flag = False
    for attr in agg_attrs:
        agg_attr_id = id(getattr(agg_obj, attr))
        for collab in all_collborators:
            collab_attr_id = id(getattr(collab, attr))
            if agg_attr_id is collab_attr_id:
                attr_ref_flag = True
                mis_matched_ref.get(collab).append(attr)

    if attr_ref_flag:
        print(
            f"{bcolors.FAIL}...Aggregator references are shared between collaborators."
            + f"{bcolors.ENDC}"
        )
    else:
        print(
            f"{bcolors.OKGREEN}  Pass : Reference test passed for aggregator.{bcolors.ENDC}"
        )


if __name__ == "__main__":
    # Setup participants
    aggregator = Aggregator()

    # Setup collaborators private attributes via callable function
    collaborator_names = ["Portland", "Seattle", "Chandler", "Bangalore"]

    def callable_to_initialize_collaborator_private_attributes(index):
        return {"index": index + 1}

    collaborators = []
    for idx, collaborator_name in enumerate(collaborator_names):
        collaborators.append(
            Collaborator(
                name=collaborator_name,
                private_attributes_callable=callable_to_initialize_collaborator_private_attributes,
                index=idx,
            )
        )

    local_runtime = LocalRuntime(
        aggregator=aggregator,
        collaborators=collaborators,
    )

    if len(sys.argv) > 1:
        if sys.argv[1] == "ray":
            local_runtime = LocalRuntime(
                aggregator=aggregator, collaborators=collaborators, backend="ray"
            )

    print(f"Local runtime collaborators = {local_runtime.collaborators}")

    testflow = TestFlowReference(checkpoint=True)
    testflow.runtime = local_runtime

    for i in range(2):
        print(f"Starting round {i}...")
        testflow.run()
