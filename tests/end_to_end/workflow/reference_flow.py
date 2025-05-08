# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from tests.end_to_end.utils.exceptions import ReferenceFlowException

import io
import math
import logging
import torch.nn as nn
import torch.optim as optim
import inspect
from types import MethodType

log = logging.getLogger(__name__)


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
    Testflow to validate
    - Whether aggregator attributes are modified in collaborator steps, AND
    - Whether collaborator attributes are unique
    """

    __test__ = False  # to prevent pytest from trying to discover tests in the class

    @aggregator
    def start(self):
        """
        Flow start.

        """
        log.info("Testing FederatedFlow - Starting Test for validating references.")
        self.next(self.test_create_agg_attr)

    @aggregator
    def test_create_agg_attr(self):
        """
        Create different types of attributes.
        """
        self.agg_attr_int = 10
        self.agg_attr_str = "Test string data"
        self.agg_attr_list = [1, 2, 5, 6, 7, 8]
        self.agg_attr_dict = {key: key for key in range(5)}
        self.agg_attr_math = math.sqrt(2)
        self.agg_attr_complex_num = complex(2, 3)

        self.collaborators = self.runtime.collaborators

        # Store aggregator attributes for validation in join step
        self.agg_attr_id_store = {}
        self.agg_attr_val_store = {}
        agg_attr_list = filter_attrs(inspect.getmembers(self))
        for attr in agg_attr_list:
            self.agg_attr_id_store[attr] = id(getattr(self, attr))
            self.agg_attr_val_store[attr] = getattr(self, attr)

        self.next(
            self.test_create_collab_attr,
            foreach="collaborators",
            exclude=["agg_attr_val_store", "agg_attr_id_store"],
        )

    @collaborator
    def test_create_collab_attr(self):
        """
        Modify the attributes of aggregator
        Create different types of collaborator attributes
        """
        self.agg_attr_int += self.index
        self.agg_attr_str = self.agg_attr_str + " " + self.input
        self.agg_attr_list.append(self.index)
        self.agg_attr_dict.update({f"{self.index}": self.index})
        self.agg_attr_math += self.index
        self.agg_attr_complex_num += complex(self.index, self.index)
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

        self.collab_attr_model = Net()
        self.collab_attr_optimizer = optim.SGD(
            self.collab_attr_model.parameters(), lr=1e-3, momentum=1e-2
        )

        self.next(self.test_create_more_collab_attr)

    @collaborator
    def test_create_more_collab_attr(self):
        """
        Create different types of collaborator attributes.
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

        self.next(
            self.join,
            include=[
                "collab_attr_int_one",
                "collab_attr_str_one",
                "collab_attr_list_one",
                "collab_attr_dict_one",
                "collab_attr_file_one",
                "collab_attr_math_one",
                "collab_attr_complex_num_one",
                "collab_attr_log_one",
                "collab_attr_model",
                "collab_attr_optimizer",
                "collab_attr_int_two",
                "collab_attr_str_two",
                "collab_attr_list_two",
                "collab_attr_dict_two",
                "collab_attr_file_two",
                "collab_attr_math_two",
                "collab_attr_complex_num_two",
                "collab_attr_log_two"
            ],
        )

    @aggregator
    def join(self, inputs):
        """
        Validate attributes
        """
        # Validate aggregator attribute are not modified in collaborator steps
        agg_validation_result = validate_agg_attr_ref(self)

        # Validate collaborators are not sharing attributes
        col_validation_result = validate_collab_attr_ref(inputs)

        assert (
            agg_validation_result and col_validation_result
        ), f" ... Testflow Reference failed"

        self.next(self.end)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """
        log.info("Testing FederatedFlow - Ending test for validating the references.")


def filter_attrs(attr_list):
    """
    Filters a list of attribute tuples to return only valid attribute names.

    An attribute is considered valid if:
    - It does not start with an underscore.
    - It is not in the list of reserved words: ["checkpoint", "execute_next", "execute_task_args", "collaborators", "runtime"].
    - It is not an attribute of the TestFlowReference class.
    - It is not an instance of MethodType.

    Args:
        attr_list (list): A list of tuples where each tuple contains an attribute name and its value.

    Returns:
        list: A list of valid attribute names.
    """
    valid_attrs = []
    reserved_words = [
        "checkpoint",
        "execute_next",
        "execute_task_args",
        "collaborators",
        "runtime",
    ]
    for attr in attr_list:
        if not attr[0].startswith("_") and attr[0] not in reserved_words:
            if not isinstance(attr[1], MethodType):
                valid_attrs.append(attr[0])
    return valid_attrs


def validate_agg_attr_ref(agg_obj):
    """
    Verifies aggregator attributes are not modified after collaborator execution
    """
    agg_attrs = filter_attrs(inspect.getmembers(agg_obj))
    agg_attrs.remove("agg_attr_val_store")
    agg_attrs.remove("agg_attr_id_store")
    validation = True
    for attr in agg_attrs:
        if agg_obj.agg_attr_val_store.get(attr) != getattr(agg_obj, attr):
            validation = False
            print(f"FAILED. Aggregator attribute {attr} is modified")
            print(
                f"...VALUE of {attr}: {agg_obj.agg_attr_val_store.get(attr)} != {getattr(agg_obj, attr)}"
            )
            print(
                f"...ID of {attr}: {agg_obj.agg_attr_id_store.get(attr)} != {id(getattr(agg_obj, attr))}"
            )

    return validation


def validate_collab_attr_ref(collab_obj_list):
    """
    Verifies collaborators attributes identities are unique
    """
    collab_attr_list = filter_attrs(inspect.getmembers(collab_obj_list[0]))
    validation = True
    for attr_name in collab_attr_list:
        for idx, cur_collab_obj in enumerate(collab_obj_list):
            for next_colab_obj in collab_obj_list[idx + 1 :]:
                if id(getattr(cur_collab_obj, attr_name)) == id(
                    getattr(next_colab_obj, attr_name)
                ):
                    validation = False
                    log.info(
                        f"FAILED. Identity matched between {cur_collab_obj.input} and {next_colab_obj.input} for {attr_name}"
                    )

    return validation
