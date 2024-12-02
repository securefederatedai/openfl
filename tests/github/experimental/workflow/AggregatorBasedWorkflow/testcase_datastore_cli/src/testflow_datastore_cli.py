# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator

batch_size_train = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


class Bcolors:
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
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc1 = nn.Linear(1440, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1440)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x)


def inference(network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    accuracy = float(correct / len(test_loader.dataset))
    return accuracy


class TestFlowDatastoreAndCli(FLSpec):
    """
    Testflow for Dataflow and CLI Functionality
    """
    def __init__(self, model=None, optimizer=None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            self.optimizer = optimizer
        else:
            self.model = Net()
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=momentum
            )
        self.num_rounds = rounds
        self.current_round = 0

    @aggregator
    def start(self):
        print(
            "Testing FederatedFlow - Starting Test for Dataflow and CLI Functionality"
        )
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.next(
            self.aggregated_model_validation,
            foreach="collaborators",
            exclude=["private"],
        )

    @collaborator
    def aggregated_model_validation(self):
        print("Performing aggregated model validation for collaborator")
        self.agg_validation_score = inference(self.model, self.test_loader)
        self.next(self.train)

    @collaborator
    def train(self):
        print("Train the model")
        self.model.train()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum
        )
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                self.loss = loss.item()
                torch.save(self.model.state_dict(), "model.pth")
                torch.save(self.optimizer.state_dict(), "optimizer.pth")
        self.training_completed = True
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = inference(self.model, self.test_loader)
        print("Doing local model validation for collaborator")
        self.next(self.join, exclude=["training_completed"])

    @aggregator
    def join(self, inputs):
        print("Executing join")
        self.current_round += 1
        if self.current_round < self.num_rounds:
            self.next(self.start)
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print("This is the end of the flow")

        expected_flow_steps = [
            "start",
            "aggregated_model_validation",
            "train",
            "local_model_validation",
            "join",
        ]  # List to verify expected steps
        validate_datastore_cli(
            self, expected_flow_steps, self.num_rounds
        )  # Function to validate datastore and cli


def validate_datastore_cli(flow_obj, expected_flow_steps, num_rounds):
    """
    This function test the flow as below
    1. Verify datastore steps and expected steps are matching
    2. Verify task stdout and task stderr verified through \
        cli is as expected
    3. Verify no of tasks executed is aligned with the total \
        number of rounds and total number of collaborators
    """
    validate_flow_error = []

    verify_stdout = {
        "start":
            "\x1b[94mTesting FederatedFlow - Starting Test for Dataflow"
            + " and CLI Functionality\x1b[0m\x1b[94m\n\x1b[0m\n",
        "aggregated_model_validation":
            "\x1b[94mPerforming aggregated model validation for"
            + " collaborator\x1b[0m\x1b[94m\n\x1b[0m\n",
        "train": "\x1b[94mTrain the model\x1b[0m\x1b[94m\n\x1b[0m\n",
        "local_model_validation":
            "\x1b[94mDoing local model validation for collaborator"
            + "\x1b[0m\x1b[94m\n\x1b[0m\n",
        "join": "\x1b[94mExecuting join\x1b[0m\x1b[94m\n\x1b[0m\n",
        "end": "\x1b[94mThis is the end of the flow\x1b[0m\x1b[94m\n\x1b[0m\n",
    }

    # fetch data from metaflow
    from metaflow import Flow

    cli_flow_obj = Flow("TestFlowDatastoreAndCli")
    cli_flow_steps = list(list(cli_flow_obj)[0])
    cli_step_names = [step.id for step in cli_flow_steps]

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
            f"{Bcolors.FAIL}... Error : Number of steps fetched from \
                Datastore through CLI do not match the Expected steps provided {Bcolors.ENDC}  \n"
        )

    if len(missing_steps_in_cli) != 0:
        validate_flow_error.append(
            f"{Bcolors.FAIL}... Error : Following steps missing from Datastore: \
                {missing_steps_in_cli} {Bcolors.ENDC}  \n"
        )

    if len(extra_steps_in_cli) != 0:
        validate_flow_error.append(
            f"{Bcolors.FAIL}... Error : Following steps are extra in Datastore: \
                {extra_steps_in_cli} {Bcolors.ENDC}  \n"
        )

    for step in cli_flow_steps:
        task_count = 0
        func = getattr(flow_obj, step.id)
        for task in list(step):
            task_count = task_count + 1
            if verify_stdout.get(step.id) != task.stdout:
                validate_flow_error.append(
                    f"{Bcolors.FAIL}... Error : task stdout detected issues : \
                        {step} {task} {Bcolors.ENDC} \n"
                )

        if (
            (func.aggregator_step)
            and (task_count != num_rounds)
            and (func.__name__ != "end")
        ):
            validate_flow_error.append(
                f"{Bcolors.FAIL}... Error : More than one execution detected \
                    for Aggregator Step: {step} {Bcolors.ENDC} \n"
            )

        if (
            (func.aggregator_step)
            and (task_count != 1)
            and (func.__name__ == "end")
        ):
            validate_flow_error.append(
                f"{Bcolors.FAIL}... Error : More than one execution detected \
                    for Aggregator Step: {step} {Bcolors.ENDC} \n"
            )

        if (func.collaborator_step) and (
            task_count != len(flow_obj.collaborators) * num_rounds
        ):
            validate_flow_error.append(
                f"{Bcolors.FAIL}... Error : Incorrect number of execution \
                    detected for Collaborator Step: {step}. \
                        Expected: {num_rounds*len(flow_obj.collaborators)} \
                        Actual: {task_count}{Bcolors.ENDC} \n"
            )

    if validate_flow_error:
        display_validate_errors(validate_flow_error)
    else:
        print(f"""{Bcolors.OKGREEN}\n**** Summary of internal flow testing ****
              No issues found and below are the tests that ran successfully
              1. Datastore steps and expected steps are matching
              2. Task stdout and task stderr verified through metaflow cli is as expected
              3. Number of tasks are aligned with number of rounds and number """
              f"""of collaborators {Bcolors.ENDC}""")


def display_validate_errors(validate_flow_error):
    """
    Function to display error that is captured during datastore and cli test
    """
    print(
        f"{Bcolors.OKBLUE}Testing FederatedFlow - Ending test for validatng \
        the Datastore and Cli Testing {Bcolors.ENDC}"
    )
    print("".join(validate_flow_error))
    print(f"{Bcolors.FAIL}\n ... Test case failed ...  {Bcolors.ENDC}")
