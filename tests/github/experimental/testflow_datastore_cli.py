import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np

from copy import deepcopy

from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

mnist_train = torchvision.datasets.MNIST(
    "files/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_test = torchvision.datasets.MNIST(
    "files/",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)


class bcolors:
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


class TestFlow_datastore_and_cli(FLSpec):
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
            f"Testing FederatedFlow - Starting Test for Dataflow and CLI Functionality"
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
        print(f"Performing aggregated model validation for collaborator")
        self.agg_validation_score = inference(self.model, self.test_loader)
        self.next(self.train)

    @collaborator
    def train(self):
        print("Train the model")
        self.model.train()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum
        )
        train_losses = []
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
        print(f"Doing local model validation for collaborator")
        self.next(self.join, exclude=["training_completed"])

    @aggregator
    def join(self, inputs):
        print(f"Executing join")
        self.current_round += 1
        if self.current_round < self.num_rounds:
            self.next(self.start)
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(f"This is the end of the flow")


def validate_datastore_cli(flow_obj, expected_flow_steps, num_rounds):
    """
    This function test the flow as below

    1. Verify datastore steps and expected steps are matching
    2. Verify task stdout and task stderr verified through cli is as expected
    3. Verify no of tasks executed is aligned with the total number of rounds and total number of collaborators
    """
    validate_flow_error = []

    verify_stdout = {
        "start": "Testing FederatedFlow - Starting Test for Dataflow and CLI Functionality\n",
        "aggregated_model_validation": "Performing aggregated model validation for collaborator\n",
        "train": "Train the model\n",
        "local_model_validation": "Doing local model validation for collaborator\n",
        "join": "Executing join\n",
        "end": "This is the end of the flow\n",
    }

    verify_stderr = {
        "start": "",
        "local_model_validation": "",
        "train": "",
        "join": "",
        "end": "",
        "aggregated_model_validation": "",
    }

    # fetch data from metaflow
    from metaflow import Flow

    cli_flow_obj = Flow("TestFlow_datastore_and_cli")
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
            f"{bcolors.FAIL}... Error : Number of steps fetched from Datastore through CLI do not match the Expected steps provided {bcolors.ENDC}  \n"
        )

    if len(missing_steps_in_cli) != 0:
        validate_flow_error.append(
            f"{bcolors.FAIL}... Error : Following steps missing from Datastore: {missing_steps_in_cli} {bcolors.ENDC}  \n"
        )

    if len(extra_steps_in_cli) != 0:
        validate_flow_error.append(
            f"{bcolors.FAIL}... Error : Following steps are extra in Datastore: {extra_steps_in_cli} {bcolors.ENDC}  \n"
        )

    for step in cli_flow_steps:
        task_count = 0
        func = getattr(flow_obj, step.id)
        for task in list(step):
            task_count = task_count + 1
            if verify_stdout.get(step.id) != task.stdout:
                validate_flow_error.append(
                    f"{bcolors.FAIL}... Error : task stdout detected issues : {step} {task} {bcolors.ENDC} \n"
                )

            # if (
            #     step.id == "start_and_init_collab"
            #     or step.id == "start_and_init_aggregator"
            # ):
            #     if not verify_stderr.get(step.id) in task.stderr:
            #         validate_flow_error.append(
            #             f"{bcolors.FAIL}... Error : task stderr detected issues : {step} {task} {bcolors.ENDC} \n"
            #         )
            # elif not verify_stderr.get(step.id) == task.stderr:
            #     validate_flow_error.append(
            #         f"{bcolors.FAIL}... Error : task stderr detected issues : {step} {task} {bcolors.ENDC} \n"
            #     )

        if (
            (func.aggregator_step == True)
            and (task_count != num_rounds)
            and (func.__name__ != "end")
        ):
            validate_flow_error.append(
                f"{bcolors.FAIL}... Error : More than one execution detected for Aggregator Step: {step} {bcolors.ENDC} \n"
            )

        if (
            (func.aggregator_step == True)
            and (task_count != 1)
            and (func.__name__ == "end")
        ):
            validate_flow_error.append(
                f"{bcolors.FAIL}... Error : More than one execution detected for Aggregator Step: {step} {bcolors.ENDC} \n"
            )

        if (func.collaborator_step == True) and (
            task_count != len(flow_obj.collaborators) * num_rounds
        ):
            validate_flow_error.append(
                f"{bcolors.FAIL}... Error : Incorrect number of execution detected for Collaborator Step: {step}. Expected: {num_rounds*len(flow_obj.collaborators)} Actual: {task_count}{bcolors.ENDC} \n"
            )

    if validate_flow_error:
        display_validate_errors(validate_flow_error)
    else:
        print(
            f"""{bcolors.OKGREEN}\n **** Summary of internal flow testing ****
        No issues found and below are the tests that ran successfully 
        1. Datastore steps and expected steps are matching
        2. Task stdout and task stderr verified through metaflow cli is as expected
        3. Number of tasks are aligned with number of rounds and number of collaborators {bcolors.ENDC}"""
        )


def display_validate_errors(validate_flow_error):
    """
    Function to display error that is captured during datastore and cli test
    """
    print(
        f"{bcolors.OKBLUE}Testing FederatedFlow - Ending test for validatng the Datastore and Cli Testing {bcolors.ENDC}"
    )
    print("".join(validate_flow_error))
    print(f"{bcolors.FAIL}\n ... Test case failed ...  {bcolors.ENDC}")


if __name__ == "__main__":
    # Setup participants
    aggregator = Aggregator()
    aggregator.private_attributes = {}

    # Setup collaborators with private attributes
    collaborator_names = ["Portland", "Seattle", "Chandler", "Bangalore"]
    collaborators = [Collaborator(name=name) for name in collaborator_names]

    for idx, collaborator in enumerate(collaborators):
        local_train = deepcopy(mnist_train)
        local_test = deepcopy(mnist_test)
        local_train.data = mnist_train.data[idx :: len(collaborators)]
        local_train.targets = mnist_train.targets[idx :: len(collaborators)]
        local_test.data = mnist_test.data[idx :: len(collaborators)]
        local_test.targets = mnist_test.targets[idx :: len(collaborators)]
        collaborator.private_attributes = {
            "train_loader": torch.utils.data.DataLoader(
                local_train, batch_size=batch_size_train, shuffle=True
            ),
            "test_loader": torch.utils.data.DataLoader(
                local_test, batch_size=batch_size_train, shuffle=True
            ),
        }

    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators, backend="ray"
    )
    print(f"Local runtime collaborators = {local_runtime.collaborators}")
    num_rounds = 5
    model = None
    optimizer = None
    flflow = TestFlow_datastore_and_cli(model, optimizer, num_rounds, checkpoint=True)
    flflow.runtime = local_runtime
    flflow.run()

    expected_flow_steps = [
        "start",
        "aggregated_model_validation",
        "train",
        "local_model_validation",
        "join",
        "end",
    ]  # List to verify expected steps
    validate_datastore_cli(
        flflow, expected_flow_steps, num_rounds
    )  # Function to validate datastore and cli
