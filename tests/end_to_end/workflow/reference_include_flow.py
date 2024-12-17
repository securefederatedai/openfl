# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

pass
import torch.nn as nn
import torch.optim as optim
import inspect
from types import MethodType
import logging

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MIN_COLLECTION_COUNT = 2


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

    step_one_collab_attrs = []
    step_two_collab_attrs = []

    @aggregator
    def start(self):
        self.agg_agg_attr_dict = {key: key for key in range(5)}
        log.info("Testing FederatedFlow - Starting Test for validating references")
        self.next(self.test_create_agg_attr, include=["agg_agg_attr_dict"])

    @aggregator
    def test_create_agg_attr(self):
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
        self.collab_attr_list_one = [1, 2, 5, 6, 7, 8]
        self.collab_attr_dict_one = {key: key for key in range(5)}

        TestFlowReferenceWithInclude.step_one_collab_attrs.append(self)

        if (
            len(TestFlowReferenceWithInclude.step_one_collab_attrs)
            >= MIN_COLLECTION_COUNT
        ):
            collab_attr_list = filter_attrs(inspect.getmembers(self))
            matched_ref_dict = find_matched_references(
                collab_attr_list,
                TestFlowReferenceWithInclude.step_one_collab_attrs,
            )
            validate_references(matched_ref_dict)

        self.next(self.test_create_more_collab_attr, include=["collab_attr_dict_one"])

    @collaborator
    def test_create_more_collab_attr(self):
        self.collab_attr_list_two = [1, 2, 3, 5, 6, 8]
        self.collab_attr_dict_two = {key: key for key in range(5)}

        TestFlowReferenceWithInclude.step_two_collab_attrs.append(self)

        if (
            len(TestFlowReferenceWithInclude.step_two_collab_attrs)
            >= MIN_COLLECTION_COUNT
        ):
            collab_attr_list = filter_attrs(inspect.getmembers(self))
            matched_ref_dict = find_matched_references(
                collab_attr_list,
                TestFlowReferenceWithInclude.step_two_collab_attrs,
            )
            validate_references(matched_ref_dict)

        self.next(self.join, include=["collab_attr_dict_two"])

    @aggregator
    def join(self, inputs):
        all_attr_list = filter_attrs(inspect.getmembers(inputs[0]))

        matched_ref_dict = find_matched_references(all_attr_list, inputs)
        validate_references(matched_ref_dict)
        all_shared_attr = ""
        log.info("Reference test summary:")
        for val in matched_ref_dict.values():
            all_shared_attr = all_shared_attr + ",".join(val)
        if all_shared_attr:
            log.error(f"Test case failed for {all_shared_attr}")
        else:
            log.info("Test case passed for all the attributes.")
        self.next(self.end)

    @aggregator
    def end(self):
        log.info("Testing FederatedFlow - Ending test for validating the references.")
        TestFlowReferenceWithInclude.step_one_collab_attrs = []
        TestFlowReferenceWithInclude.step_two_collab_attrs = []


def filter_attrs(attr_list):
    valid_attrs = []
    reserved_words = ["next", "runtime", "execute_next"]
    for attr in attr_list:
        if (
            not attr[0].startswith("_")
            and attr[0] not in reserved words
            and not hasattr(TestFlowReferenceWithInclude, attr[0])
        ):
            if not isinstance(attr[1], MethodType):
                valid_attrs.append(attr[0])
    return valid_attrs


def find_matched_references(collab_attr_list, all_collaborators):
    matched_ref_dict = {}
    for i in range(len(all_collaborators)):
        matched_ref_dict[all_collaborators[i].input] = []

    for attr_name in collab_attr_list:
        for i, curr_collab in enumerate(all_collaborators):
            for next_collab in all_collaborators[i + 1:]:
                if hasattr(curr_collab, attr_name) and hasattr(next_collab, attr_name):
                    if getattr(curr_collab, attr_name) is getattr(
                        next_collab, attr_name
                    ):
                        matched_ref_dict[curr_collab.input].append(attr_name)
                        log.error(
                            f"Reference test failed - {curr_collab.input} sharing same "
                            + f"{attr_name} reference with {next_collab.input}"
                        )

    return matched_ref_dict


def validate_references(matched_ref_dict):
    collborators_sharing_ref = []
    reference_flag = False

    for collab, val in matched_ref_dict.items():
        if val:
            collborators_sharing_ref.append(collab)
            reference_flag = True

    assert not reference_flag, "Reference test failed"
