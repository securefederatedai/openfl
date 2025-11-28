# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Co-authored-by: Anindya S. Paul <anindya.s.paul@intel.com>
# Co-authored-by: Brandon Edwards <brandon.edwards@intel.com>
# Co-authored-by: Mansi Sharma <mansi.sharma@intel.com>

from clip_optimizer import ClipOptimizer
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np
from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from openfl.experimental.workflow.placement import aggregator, collaborator

from torch.distributions.normal import Normal
from opacus.accountants.rdp import RDPAccountant
from opacus.data_loader import DPDataLoader
from torch.utils.data import DataLoader

import argparse
import yaml

import warnings

warnings.filterwarnings("ignore")


batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# Fixing the seed for result repeatation: remove below to stop repeatable runs
# ----------------------------------
random_seed = 5495300300540669060
g_device = torch.Generator(device="cuda")
# Uncomment the line below to use g_cpu if not using cuda
# g_device = torch.Generator() # noqa: E800
# NOTE: remove below to stop repeatable runs
g_device.manual_seed(random_seed)
print(f"\n\nWe are using seed: {random_seed}")
# ----------------------------------

# Loading torchvision MNIST datasets
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def default_optimizer(model):
    """
    Return a new optimizer: we have only tested torch.optim.SGD w/ momentum
    however, we encouraging users to test other optimizers (i.e. torch.optim.Adam)
    and provide us feedback regarding your observations.

    Args:
        model:   NN model architected from nn.module class
    """
    return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def FedAvg(models, previous_global_model=None, dp_params=None):  # NOQA: N802
    """
    Return a Federated average model based on Fedavg algorithm: H. B. Mcmahan,
    E. Moore, D. Ramage, S. Hampson, and B. A. Y.Arcas,
    “Communication-efficient learning of deep networks
    from decentralized data,” 2017.
    This tutorial utilizes non-weighted averaging of collaborator
    model updates regardless of whether DP config is used.
    Weighted FedAvg is currently not supported.

    Args:
        models: Python list of locally trained models by each collaborator
        at the current round
        previous_global_model: Federated averaged model from the previous round
        dp_params: Python dictionary for differential privacy
        specific hyperparameters as read from "test_config.yml"
    """
    if dp_params is not None and previous_global_model is not None:
        # Validate that in fact the local models clipped their updates
        non_delta_states = [model.state_dict() for model in models]
        previous_global_model_state = previous_global_model.cpu().state_dict()
        delta_states = []
        for non_delta_state in non_delta_states:
            delta_states.append(
                {
                    key: non_delta_state[key] - previous_global_model_state[key]
                    for key in non_delta_state
                }
            )
        for idx, state in enumerate(delta_states):
            per_layer_norms = []
            for key, tensor in state.items():
                per_layer_norms.append(torch.norm(tensor))

            if torch.norm(torch.Tensor(per_layer_norms)) > dp_params["clip_norm"]:
                raise ValueError(
                    f"The model with index {idx} had update whose "
                    + "L2-norm was greater than clip norm."
                    + "Correct the local periodic clipping."
                )
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = models[0].state_dict()
    if len(state_dicts) > 1:
        for key in models[0].state_dict():
            state_dict[key] = np.sum(
                np.array([state[key] for state in state_dicts], dtype=object), axis=0
            ) / len(models)
    new_model.load_state_dict(state_dict)
    return new_model


def inference(network, test_loader, device):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Avg. loss: {test_loss:.4f},"
        + f"Accuracy: {correct}/{len(test_loader.dataset)}"
        + f"({(100.0 * correct / len(test_loader.dataset)):.0f})\n"
    )
    accuracy = float(correct / len(test_loader.dataset))
    return accuracy


def optimizer_to_device(optimizer, device):
    """
    Sending the "torch.optim.Optimizer" object into the specified device
    for model training and inference

    Args:
        optimizer: torch.optim.Optimizer from "default_optimizer" function
        device: CUDA device id or "cpu"
    """
    if optimizer.state_dict != {}:
        if isinstance(optimizer, optim.SGD):
            for param in optimizer.param_groups[0]["params"]:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        elif isinstance(optimizer, optim.Adam):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    else:
        raise (
            ValueError("Current optimizer state does not have dict keys: please verify")
        )


def clip_testing_on_optimizer_parameters(
    optimizer_before_step_params,
    optimizer_after_step_params,
    collaborator_name,
    round_num,
    device,
):
    """
    Test to check that optimizer parameters are clipped after performing
    optimizer step method.

    Args:
        optimizer_before_step_params: optimizer parameters before step
        optimizer_after_step_params: optimizer parameters after step
        collaborator_name: name of the collaborator (Type:string)
        round_num: current round (Type:int)
        device: CUDA device id or "cpu"
    """
    len_equal_tensor = 0
    for param_idx in range(len(optimizer_after_step_params)):
        for tensor_1, tensor_2 in zip(
            optimizer_before_step_params[param_idx],
            optimizer_after_step_params[param_idx],
        ):
            if torch.equal(tensor_1.to(device), tensor_2.to(device)) is True:
                len_equal_tensor += 1
    if len_equal_tensor == len(optimizer_after_step_params):
        raise (
            ValueError(
                "No clipping effect: Optimizer param data is the same "
                + "between before and after optimizer step for collaborator: "
                + f"{collaborator_name} at round {round_num}"
            )
        )


def validate_dp_params(dp_params):
    """
    The differential privacy block should have the exact keys as provided below.

    Args:
        dp_params: Python dictionary for differential privacy
        specific hyperparameters as read from "test_config.yml"
    """
    required_dp_keys = [
        "clip_norm",
        "noise_multiplier",
        "delta",
        "sample_rate",
        "clip_frequency",
    ]
    keys = dp_params.keys()
    excess = list(set(keys).difference(set(required_dp_keys)))
    deficit = list(set(required_dp_keys).difference(keys))

    if excess != []:
        print(
            f"\nCAUTION: The keys: {excess} where provided in the 'differential_privacy'"
            + "block of the flow config and are not being used.\n"
        )
    if deficit != []:
        raise ValueError(
            f"The 'differential_privacy' block is missing the required keys: {deficit}"
        )


def parse_config(config_path):
    """
    Parse "test_config.yml".

    Args:
        config_path: Path of "test_config.yml"
    """
    with open(config_path, "rb") as _file:
        config = yaml.safe_load(_file)
    return config


def add_noise_on_aggegated_parameters(collaborators, model, dp_params):
    """
    Adds noise on aggregated model parameters performed at the aggregator.

    Args:
        collaborators: Python list of collaborator name strings
        model: Federeated averaged model
        dp_params: Python dictionary for differential privacy
        specific hyperparameters as read from "test_config.yml"
    """
    state_dict = model.state_dict()
    normal_distribution = Normal(
        loc=0, scale=dp_params["noise_multiplier"] * dp_params["clip_norm"]
    )
    with torch.no_grad():
        for param_tensor in model.state_dict():
            noise_samples = normal_distribution.sample(
                state_dict[param_tensor].shape
            ) / int(dp_params["sample_rate"] * float(len(collaborators)))
            state_dict[param_tensor].add_(noise_samples)
    model.load_state_dict(state_dict)
    return model


class FederatedFlow(FLSpec):
    def __init__(
        self,
        config_path,
        model,
        collaborator_names,
        device,
        total_rounds=10,
        top_model_accuracy=0,
        clip_test=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.previous_global_model = Net()
        self.collaborator_names = collaborator_names
        self.total_rounds = total_rounds
        self.top_model_accuracy = top_model_accuracy
        self.device = device
        self.clip_test = clip_test
        self.round = 0  # starting round
        self.aggregated_model_accuracy = None
        self.privacy_accountant = RDPAccountant()
        config = parse_config(config_path)

        if "differential_privacy" not in config:
            self.dp_params = None
        else:
            self.dp_params = config["differential_privacy"]
            print(f"Here are dp_params: {self.dp_params}")
            validate_dp_params(self.dp_params)

    @aggregator
    def start(self):
        print("Performing initialization for model")
        self.collaborators = self.runtime.collaborators
        self.private = 10

        if self.dp_params is None:
            self.round_collaborators = self.collaborators
            self.optimizers = {
                collaborator_name: default_optimizer(self.model)
                for collaborator_name in self.collaborator_names
            }

        else:
            self.optimizers = {
                collaborator_name: ClipOptimizer(
                    base_optimizer=default_optimizer(self.model),
                    device=self.device,
                    clip_norm=self.dp_params["clip_norm"],
                    clip_freq=self.dp_params["clip_frequency"],
                )
                for collaborator_name in self.collaborator_names
            }
            self.sample_rate = self.dp_params["sample_rate"]
            global_data_loader = DataLoader(
                self.collaborators,
                batch_size=int(self.sample_rate * float(len(self.collaborators))),
            )
            dp_data_loader = DPDataLoader.from_data_loader(
                global_data_loader, distributed=False
            )
            collaborator_batch = []
            batch_sizes = []
            for cols in dp_data_loader:
                batch_sizes.append(len(cols))
                collaborator_batch.append(cols)
            self.round_collaborators = collaborator_batch[0]

        if len(self.round_collaborators) != 0:
            if not isinstance(self.round_collaborators[0], torch.Tensor):
                print(20 * "#")
                print(f"Round {self.round}...")
                print("Batch sizes sampled:", batch_sizes)
                print(
                    f"Collaborators patricipated in Round: {self.round}",
                    self.round_collaborators,
                )
                print(20 * "#")
                self.next(
                    self.aggregated_model_validation,
                    foreach="round_collaborators",
                    exclude=["private"],
                )
            else:
                print(f"No collaborator selected for training at Round: {self.round}")
                self.next(self.check_round_completion)
        else:
            print(f"No collaborator selected for training at Round: {self.round}")
            self.next(self.check_round_completion)

    @collaborator
    def aggregated_model_validation(self):
        print(f"Performing aggregated model validation for collaborator {self.input}")
        self.model = self.model.to(self.device)
        self.previous_global_model = self.previous_global_model.to(self.device)

        # verifying that model went to the correct GPU device
        assert next(self.model.parameters()).device == self.device
        assert next(self.previous_global_model.parameters()).device == self.device

        self.agg_validation_score = inference(self.model, self.test_loader, self.device)
        print(f"{self.input} value of {self.agg_validation_score}")
        self.collaborator_name = self.input
        self.next(self.train)

    @collaborator
    def train(self):
        print(f"Performing model training for collaborator {self.input}")
        self.optimizer = ClipOptimizer(
            base_optimizer=default_optimizer(self.model),
            device=self.device,
            clip_norm=self.dp_params["clip_norm"],
            clip_freq=self.dp_params["clip_frequency"],
        )

        # copy state dict from optimizer object from previous round to the newly
        # instantiated optimizer for the same collaborator
        if self.round > 0:
            optimizer_to_device(optimizer=self.optimizer, device=self.device)
            self.optimizer.load_state_dict(
                deepcopy(self.optimizers[self.input].state_dict())
            )

        self.model.train()
        train_losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target).to(self.device)
            loss.backward()

            if self.clip_test:
                optimizer_before_step_params = [
                    param.data for param in self.optimizer.param_groups()[0]["params"]
                ]

            self.optimizer.step(
                global_model_state=self.previous_global_model.state_dict(),
                last_iter=(batch_idx == (len(self.train_loader) - 1)),
            )

            if batch_idx % self.dp_params["clip_frequency"] == 0 or (
                batch_idx == (len(self.train_loader) - 1)
            ):
                if self.clip_test:
                    optimizer_after_step_params = [
                        param.data
                        for param in self.optimizer.param_groups()[0]["params"]
                    ]
                    clip_testing_on_optimizer_parameters(
                        optimizer_before_step_params,
                        optimizer_after_step_params,
                        self.collaborator_name,
                        self.round,
                        self.device,
                    )

            train_losses.append(loss.item())

        self.loss = np.mean(train_losses)
        self.training_completed = True

        tmp_opt = deepcopy(self.optimizers[self.input])
        tmp_opt.load_state_dict(self.optimizer.state_dict())
        self.optimizer = tmp_opt
        torch.cuda.empty_cache()
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        print(f"Performing local model validation for collaborator {self.input}")
        self.local_validation_score = inference(
            self.model, self.test_loader, self.device
        )
        print(f"{self.input} value of {self.local_validation_score}")
        self.next(self.join, exclude=["training_completed"])

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs
        ) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs
        ) / len(inputs)
        print(
            f"Average aggregated model validation values = {self.aggregated_model_accuracy}"
        )
        print(f"Average training loss = {self.average_loss}")
        print(f"Average local model validation values = {self.local_model_accuracy}")
        if self.dp_params is not None:
            self.model = FedAvg(
                [input.model.cpu() for input in inputs],
                previous_global_model=self.previous_global_model,
                dp_params=self.dp_params,
            )
        else:
            self.model = FedAvg([input.model.cpu() for input in inputs])

        if self.dp_params is not None:
            self.model = add_noise_on_aggegated_parameters(
                self.collaborators, self.model, self.dp_params
            )
            self.privacy_accountant.step(
                noise_multiplier=self.dp_params["noise_multiplier"],
                sample_rate=self.sample_rate,
            )
            epsilon, best_alpha = self.privacy_accountant.get_privacy_spent(
                delta=self.dp_params["delta"]
            )
            print(20 * "#")
            print(
                f"\nCurrent privacy spent using delta={self.dp_params['delta']} "
                + f"is epsilon={epsilon} (best alpha was: {best_alpha})."
            )
            print(20 * "#")
        self.previous_global_model.load_state_dict(deepcopy(self.model.state_dict()))
        self.optimizers.update(
            {input.collaborator_name: input.optimizer for input in inputs}
        )
        del inputs
        self.next(self.check_round_completion)

    @aggregator
    def check_round_completion(self):
        if self.round == self.total_rounds - 1:
            self.next(self.end)
        else:
            if self.aggregated_model_accuracy is not None:
                if self.aggregated_model_accuracy > self.top_model_accuracy:
                    print(
                        f"Accuracy improved to {self.aggregated_model_accuracy} for "
                        + f"round {self.round}"
                    )
                    self.top_model_accuracy = self.aggregated_model_accuracy

            if self.dp_params is not None:
                global_data_loader = DataLoader(
                    self.collaborators,
                    batch_size=int(self.sample_rate * float(len(self.collaborators))),
                )
                dp_data_loader = DPDataLoader.from_data_loader(
                    global_data_loader, distributed=False
                )
                collaborator_batch = []
                batch_sizes = []
                for cols in dp_data_loader:
                    batch_sizes.append(len(cols))
                    collaborator_batch.append(cols)
                self.round_collaborators = collaborator_batch[0]
            else:
                self.round_collaborators = self.collaborators

            self.round += 1

            if len(self.round_collaborators) != 0:
                if not isinstance(self.round_collaborators[0], torch.Tensor):
                    print(20 * "#")
                    print(f"Round {self.round}...")
                    print("Batch sizes sampled:", batch_sizes)
                    print(
                        f"Collaborators patricipated in Round: {self.round}",
                        self.round_collaborators,
                    )
                    self.next(
                        self.aggregated_model_validation,
                        foreach="round_collaborators",
                        exclude=["private"],
                    )
                    print(20 * "#")
                else:
                    print(
                        f"No collaborator selected for training at Round: {self.round}"
                    )
                    self.next(self.check_round_completion)
            else:
                print(f"No collaborator selected for training at Round: {self.round}")
                self.next(self.check_round_completion)

    @aggregator
    def end(self):
        print(20 * "#")
        print("All rounds completed successfully")
        print(20 * "#")
        print("This is the end of the flow")
        print(20 * "#")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--config_path", help="Absolute path to the flow configuration file"
    )
    argparser.add_argument(
        "--clip_test",
        default=False,
        help="Indicate enabling of optimizer param testing before and after clip",
    )

    args = argparser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Setup participants
    # Set `num_gpus=0.09` to `num_gpus=0.0` in order to run this tutorial on CPU
    aggregator = Aggregator(num_gpus=0.09)

    # Collaborator names
    collaborator_names = [
        "Portland",
        "Seattle",
        "Bangalore",
        "Chandler",
        "Austin",
        "Hudson",
        "Penang",
        "Haifa",
        "CostaRica",
        "Guadalajara",
    ]

    def callable_to_initialize_collaborator_private_attributes(
        index, n_collaborators, batch_size, train_dataset, test_dataset
    ):
        train = deepcopy(train_dataset)
        test = deepcopy(test_dataset)
        train.data = train_dataset.data[index::n_collaborators]
        train.targets = train_dataset.targets[index::n_collaborators]
        test.data = test_dataset.data[index::n_collaborators]
        test.targets = test_dataset.targets[index::n_collaborators]

        return {
            "train_loader": torch.utils.data.DataLoader(
                train, batch_size=batch_size, shuffle=True
            ),
            "test_loader": torch.utils.data.DataLoader(
                test, batch_size=batch_size, shuffle=True
            ),
        }

    collaborators = []
    for idx, collaborator_name in enumerate(collaborator_names):
        collaborators.append(
            Collaborator(
                name=collaborator_name,
                private_attributes_callable=callable_to_initialize_collaborator_private_attributes,
                # Set `num_gpus=0.09` to `num_gpus=0.0` in order to run this tutorial on CPU
                num_cpus=0.0,
                num_gpus=0.09,  # Assuming GPU(s) is available in the machine
                index=idx,
                n_collaborators=len(collaborator_names),
                batch_size=batch_size_train,
                train_dataset=mnist_train,
                test_dataset=mnist_test,
            )
        )

    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators, backend="ray"
    )
    print(f"Local runtime collaborators = {local_runtime.collaborators}")

    top_model_accuracy = 0
    model = Net()
    total_rounds = 10

    flflow = FederatedFlow(
        config_path=args.config_path,
        model=model,
        collaborator_names=collaborator_names,
        device=device,
        total_rounds=total_rounds,
        top_model_accuracy=top_model_accuracy,
        clip_test=args.clip_test,
    )
    flflow.runtime = local_runtime
    flflow.run()
