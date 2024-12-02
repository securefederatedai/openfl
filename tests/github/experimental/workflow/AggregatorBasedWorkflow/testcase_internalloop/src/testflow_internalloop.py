# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.workflow.interface.fl_spec import FLSpec
from openfl.experimental.workflow.placement.placement import aggregator, collaborator
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


class TestFlowInternalLoop(FLSpec):
    def __init__(self, model=None, optimizer=None, rounds=None, **kwargs):
        super().__init__(**kwargs)
        self.training_rounds = rounds
        self.train_count = 0
        self.end_count = 0

    @aggregator
    def start(self):
        """
        Flow start.
        """
        print(
            f"{bcolors.OKBLUE}Testing FederatedFlow - "
            + f"Test for Internal Loops - Round: {self.train_count}"
            + f" of Training Rounds: {self.training_rounds}{bcolors.ENDC}"
        )
        self.model = np.zeros((10, 10, 10))  # Test model
        self.collaborators = self.runtime.collaborators
        self.next(self.agg_model_mean, foreach="collaborators")

    @collaborator
    def agg_model_mean(self):
        """
        Calculating the mean of the model created in start.
        """
        self.agg_mean_value = np.mean(self.model)
        print(f"<Collab>: {self.input} Mean of Agg model: {self.agg_mean_value} ")
        self.next(self.collab_model_update)

    @collaborator
    def collab_model_update(self):
        """
        Initializing the model with random numbers.
        """
        print(f"<Collab>: {self.input} Initializing the model randomly ")
        self.model = np.random.randint(1, len(self.input), (10, 10, 10))
        self.next(self.local_model_mean)

    @collaborator
    def local_model_mean(self):
        """
        Calculating the mean of the model created in train.
        """
        self.local_mean_value = np.mean(self.model)
        print(f"<Collab>: {self.input} Local mean: {self.local_mean_value} ")
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        """
        Joining inputs from collaborators
        """
        self.agg_mean = sum(input.local_mean_value for input in inputs) / len(inputs)
        print(f"Aggregated mean : {self.agg_mean}")
        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        """
        Internally Loop for training rounds
        """
        self.train_count = self.train_count + 1
        if self.training_rounds == self.train_count:
            self.next(self.end)
        else:
            self.next(self.start)

    @aggregator
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """
        self.end_count += 1
        print("This is the end of the flow")

        flflow = self
        # Flow Test Begins
        expected_flow_steps = [
            "join",
            "internal_loop",
            "agg_model_mean",
            "collab_model_update",
            "local_model_mean",
            "start",
        ]  # List to verify expected steps
        try:
            validate_flow(
                flflow, expected_flow_steps
            )  # Function to validate the internal flow
        except Exception as e:
            raise e
        # Flow Test Ends


def validate_flow(flow_obj, expected_flow_steps):
    """
    Validate:
    1. If the given training round were completed
    2. If all the steps were executed
    3. If each collaborator step was executed
    4. If end was executed once
    """
    validate_flow_error = []  # List to capture any errors in the flow

    from metaflow import Flow

    cli_flow_obj = Flow("TestFlowInternalLoop")  # Flow object from CLI
    cli_flow_steps = list(cli_flow_obj.latest_run)  # Steps from CLI
    cli_step_names = [step.id for step in cli_flow_steps]

    # 1. If the given training round were completed
    if not flow_obj.training_rounds == flow_obj.train_count:
        validate_flow_error.append(
            f"{bcolors.FAIL}... Error : Number of training completed is not equal"
            + f" to training rounds {bcolors.ENDC} \n"
        )

    for step in cli_flow_steps:
        task_count = 0
        func = getattr(flow_obj, step.id)
        for task in list(step):
            task_count = task_count + 1

        # Each aggregator step should be executed for training rounds times
        if (
            (func.aggregator_step is True)
            and (task_count != flow_obj.training_rounds)
            and (step.id != "end")
        ):
            validate_flow_error.append(
                f"{bcolors.FAIL}... Error : More than one execution detected for "
                + f"Aggregator Step: {step} {bcolors.ENDC} \n"
            )

        # Each collaborator step is executed for (training rounds)*(number of collaborator) times
        if (func.collaborator_step is True) and (
            task_count != len(flow_obj.collaborators) * flow_obj.training_rounds
        ):
            validate_flow_error.append(
                f"{bcolors.FAIL}... Error : Incorrect number of execution detected for "
                + f"Collaborator Step: {step}. Expected: "
                + f"{flow_obj.training_rounds*len(flow_obj.collaborators)} "
                + f"Actual: {task_count}{bcolors.ENDC} \n"
            )

    steps_present_in_cli = [
        step for step in expected_flow_steps if step in cli_step_names
    ]
    missing_steps_in_cli = [
        step for step in expected_flow_steps if step not in cli_step_names
    ]
    extra_steps_in_cli = [
        step for step in cli_step_names if step not in expected_flow_steps
    ]

    if len(steps_present_in_cli) != len(expected_flow_steps):
        validate_flow_error.append(
            f"{bcolors.FAIL}... Error : Number of steps fetched from Datastore through CLI do not "
            + f"match the Expected steps provided {bcolors.ENDC}  \n"
        )

    if len(missing_steps_in_cli) != 0:
        validate_flow_error.append(
            f"{bcolors.FAIL}... Error : Following steps missing from Datastore: "
            + f"{missing_steps_in_cli} {bcolors.ENDC}  \n"
        )

    if len(extra_steps_in_cli) != 0:
        validate_flow_error.append(
            f"{bcolors.FAIL}... Error : Following steps are extra in Datastore: "
            + f"{extra_steps_in_cli} {bcolors.ENDC}  \n"
        )

    if not flow_obj.end_count == 1:
        validate_flow_error.append(
            f"{bcolors.FAIL}... Error : End function called more than one time...{bcolors.ENDC}"
        )

    if validate_flow_error:
        display_validate_errors(validate_flow_error)
        raise Exception(f"{bcolors.FAIL}Test for Internal Loop FAILED")
    else:
        print(
            f"""{bcolors.OKGREEN}\n **** Summary of internal flow testing ****
        No issues found and below are the tests that ran successfully
        1. Number of training completed is equal to training rounds
        2. Cli steps and Expected steps are matching
        3. Number of tasks are aligned with number of rounds and number of collaborators
        4. End function executed one time {bcolors.ENDC}"""
        )


def display_validate_errors(validate_flow_error):
    """
    Function to display error that is captured during flow test
    """
    print("".join(validate_flow_error))
