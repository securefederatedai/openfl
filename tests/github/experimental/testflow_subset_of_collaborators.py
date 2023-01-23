# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.interface.fl_spec import FLSpec
from openfl.experimental.interface.participants import Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement.placement import aggregator, collaborator
from metaflow import Step
import random
import sys
import os
import shutil


class bcolors:  # NOQA: N801
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    HEADER = "\033[95m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ENDC = "\033[0m"


class TestFlowSubsetCollaborators(FLSpec):
    """
    Testflow to validate working of Subset Collaborators in Federated Flow.

    """

    def __init__(self, random_ints=[], **kwargs) -> None:
        super().__init__(**kwargs)
        self.random_ints = random_ints

    @aggregator
    def start(self):
        """
        Starting the flow with random subset of collaborators

        """
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - Starting Test for "
            + f"validating Subset of collaborators  {bcolors.ENDC}"
        )
        self.collaborators = self.runtime.collaborators

        # select subset of collaborators
        self.subset_collabrators = self.collaborators[
            : random.choice(self.random_ints)
        ]

        print(
            f"... Executing flow for {len(self.subset_collabrators)} collaborators out of Total: "
            + f"{len(self.collaborators)}"
        )

        self.next(self.test_valid_collaborators, foreach="subset_collabrators")

    @collaborator
    def test_valid_collaborators(self):
        """
        set the collaborator name

        """
        print("printing collaborators")
        self.collaborator_ran = self.name
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        """
        List of collaboartors ran successfully

        """
        print("inside join")
        self.collaborators_ran = [input.collaborator_ran for input in inputs]
        self.next(self.end)

    @aggregator
    def end(self):
        """
        End of the flow

        """
        print(
            f"End of the test case {TestFlowSubsetCollaborators.__name__} reached."
        )


if __name__ == "__main__":
    aggregator = Aggregator()
    aggregator.private_attributes = {}

    collaborator_names = [
        "Portland",
        "Seattle",
        "Chandler",
        "Bangalore",
        "Delhi",
        "Paris",
        "London",
        "New York",
    ]
    collaborators = []
    for name in collaborator_names:
        temp_collab_obj = Collaborator(name=name)
        temp_collab_obj.private_attributes = {"name": name}
        collaborators.append(temp_collab_obj)
        del temp_collab_obj

    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators
    )

    if len(sys.argv) > 1:
        if sys.argv[1] == 'ray':
            local_runtime = LocalRuntime(
                aggregator=aggregator, collaborators=collaborators, backend='ray'
            )

    random_ints = random.sample(
        range(1, len(collaborators) + 1), len(collaborators)
    )
    tc_pass_fail = {"passed": [], "failed": []}
    for round_num in range(len(collaborators)):
        print(f"{bcolors.OKBLUE}Starting round {round_num}...{bcolors.ENDC}")

        if os.path.exists(".metaflow"):
            shutil.rmtree(".metaflow")

        testflow_subset_collaborators = TestFlowSubsetCollaborators(
            checkpoint=True, random_ints=random_ints
        )
        testflow_subset_collaborators.runtime = local_runtime
        testflow_subset_collaborators.run()

        subset_collaborators = testflow_subset_collaborators.subset_collabrators
        collaborators_ran = testflow_subset_collaborators.collaborators_ran
        random_ints = testflow_subset_collaborators.random_ints
        random_ints.remove(len(subset_collaborators))

        step = Step(
            f"TestFlowSubsetCollaborators/{testflow_subset_collaborators._run_id}/"
            + "test_valid_collaborators"
        )

        if len(list(step)) != len(subset_collaborators):
            tc_pass_fail["failed"].append(
                f"{bcolors.FAIL}...Flow only ran for {len(list(step))} "
                + f"instead of the {len(subset_collaborators)} expected "
                + f"collaborators- Testcase Failed.{bcolors.ENDC} "
            )
        else:
            tc_pass_fail["passed"].append(
                f"{bcolors.OKGREEN}Found {len(list(step))} tasks for each of the "
                + f"{len(subset_collaborators)} collaborators - "
                + f"Testcase Passed.{bcolors.ENDC}"
            )
        passed = True
        for collaborator_name in subset_collaborators:
            if collaborator_name not in collaborators_ran:
                passed = False
                tc_pass_fail["failed"].append(
                    f"{bcolors.FAIL}...Flow did not execute for "
                    + f"collaborator {collaborator_name}"
                    + f" - Testcase Failed.{bcolors.ENDC}"
                )

        if passed:
            tc_pass_fail["passed"].append(
                f"{bcolors.OKGREEN}Flow executed for all collaborators"
                + f"- Testcase Passed.{bcolors.ENDC}"
            )
    for values in tc_pass_fail.values():
        print(*values, sep="\n")

    print(
        f"{bcolors.OKBLUE}Testing FederatedFlow - Ending test for validating "
        + f"the subset of collaborators. {bcolors.ENDC}"
    )
    if tc_pass_fail.get("failed"):
        tc_pass_fail_len = len(tc_pass_fail.get("failed"))
        raise AssertionError(
            f"{bcolors.FAIL}\n {tc_pass_fail_len} Test "
            + f"case(s) failed ... {bcolors.ENDC}"
        )
