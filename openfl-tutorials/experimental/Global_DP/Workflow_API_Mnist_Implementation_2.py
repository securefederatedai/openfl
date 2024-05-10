# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Co-authored-by: Brandon Edwards <brandon.edwards@intel.com>
# Co-authored-by: Anindya S. Paul <anindya.s.paul@intel.com>
# Co-authored-by: Mansi Sharma <mansi.sharma@intel.com>

from clip_optimizer import ClipOptimizer
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
from torch.utils.data import TensorDataset
import numpy as np
from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator

from opacus import PrivacyEngine
import argparse
import yaml

import warnings

warnings.filterwarnings("ignore")


batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 5495300300540669060

g_device = torch.Generator(device="cuda")
# Uncomment the line below to use g_cpu if not using cuda
# g_device = torch.Generator() # noqa: E800
# NOTE: remove below to stop repeatable runs
g_device.manual_seed(random_seed)
print(f"\n\nWe are using seed: {random_seed}")

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


class GlobalModelTools(object):
    """
    Class to facilitate the updating of a model whose 'per-sample' delta updates
    are provided from some out of band process and passed as torch model
    state dicts to the class. A global model object is constructed and its
    per example gradients are filled in with the local model updates. Then the
    optimizer can be stepped to affect the aggregate update (with DP if desired)
    """

    def __init__(
        self, example_model_state, global_model, collaborator_names, dp_params=None
    ):
        self.example_state = example_model_state
        self.global_model = global_model
        self.collaborator_names = collaborator_names
        self.global_optimizer = torch.optim.SGD(
            params=self.global_model.parameters(), lr=1.0
        )  # This choice of optimizer is required for correct model aggregation
        self.dp_params = dp_params
        self.privacy_engine = PrivacyEngine()
        if dp_params is None:
            # this is the expected fraction of collaborators to select each round,
            # also the indepencent probability that each
            # collaborator gets selected for a round
            sample_rate = 1.0
        else:
            sample_rate = dp_params["sample_rate"]
        self.global_data_loader = torch.utils.data.DataLoader(
            TensorDataset(
                torch.Tensor(list(range(len(self.collaborator_names)))).to(torch.int)
            ),
            batch_size=int(sample_rate * float(len(collaborator_names))),
            shuffle=True,
        )
        if dp_params is not None:
            (
                self.global_model,
                self.global_optimizer,
                self.global_data_loader,
            ) = self.privacy_engine.make_private(
                module=self.global_model,
                optimizer=self.global_optimizer,
                data_loader=self.global_data_loader,
                noise_multiplier=dp_params["noise_multiplier"],
                max_grad_norm=dp_params["clip_norm"],
            )

    def populate_model_params_and_gradients(
        self, state_for_params, states_for_gradients
    ):
        if self.dp_params is not None:
            # populate param.grad_sample with individual updates from state_dicts,
            # and populate pram.grad with a random
            # properly shaped update in order to help opacus infer that shape information
            per_layer_grad_samples = {
                key: [state[key] for state in states_for_gradients]
                for key in states_for_gradients[0].keys()
            }
            for params, name in zip(
                self.global_model.parameters(), self.example_state.keys()
            ):
                params.data = state_for_params[name]
                params.grad_sample = torch.stack(per_layer_grad_samples[name], dim=0)
                # only the shape is important below, values are not important and so we
                # use only the first state
                params.grad = states_for_gradients[0][name]


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
    # indicate the default optimizer for training the target model
    return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def FedAvg(models, global_model_tools, previous_global_state, dp_params):  # NOQA: N802
    """
    This tutorial utilizes non-weighted averaging of collaborator model updates
    regardless of whether DP config is used.
    Weighted FedAvg is currently not supported.
    """
    new_model = models[0]
    # Note: the states below are non-delta states
    non_delta_states = [model.state_dict() for model in models]
    neg_delta_states = []
    for non_delta_state in non_delta_states:
        # These will be used for the gradient, so need to be opposite the model delta
        neg_delta_states.append(
            {
                key: -non_delta_state[key] + previous_global_state[key]
                for key in non_delta_state
            }
        )
    # validate that in fact the local models clipped their updates
    print()
    for idx, state in enumerate(neg_delta_states):
        per_layer_norms = []
        for key, tensor in state.items():
            per_layer_norms.append(torch.norm(tensor, dim=()))
        delta_norm = torch.norm(torch.Tensor(per_layer_norms), dim=())
        print(f"delta_norm for idx {idx} is: {delta_norm}")
        if delta_norm > dp_params["clip_norm"]:
            raise ValueError(
                f"The model with index {idx} had update whose "
                + "L2-norm was greater than clip norm. Correct the local periodic clipping."
            )
    print()
    # Clearing state from optimizer from last round so as to not leak information
    global_model_tools.global_optimizer.zero_grad()
    global_model_tools.populate_model_params_and_gradients(
        state_for_params=previous_global_state, states_for_gradients=neg_delta_states
    )
    global_model_tools.global_optimizer.step()

    # removing the '_module.' from the beggining of all keys coming from
    # global_model_tools.global_model state dict
    new_model.load_state_dict(
        {
            key[8:]: value
            for key, value in global_model_tools.global_model.state_dict().items()
        }
    )
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
        "\nTest set: Avg. loss: {test_loss:.4f},"
        f" Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({(100.0 * correct / len(test_loader.dataset)):.0f})\n"
    )
    accuracy = float(correct / len(test_loader.dataset))
    return accuracy


def optimizer_to_device(optimizer, device):
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
    # The differential privacy block should have the exact keys provided below
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
            + " block of the flow config and are not being used.\n"
        )
    if deficit != []:
        raise ValueError(
            f"The 'differential_privacy' block is missing the required keys: {deficit}"
        )


def parse_config(config_path):
    with open(config_path, "rb") as _file:
        config = yaml.safe_load(_file)
    return config


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
        self.global_model = Net()
        self.collaborator_names = collaborator_names
        self.total_rounds = total_rounds
        self.round = None
        self.top_model_accuracy = top_model_accuracy
        self.device = device
        self.clip_test = clip_test
        # we will set this attribute at the beginning of each round (tracks using
        # indices against the collaborator list)
        self.round_collaborator_idxs = None

        config = parse_config(config_path)

        if "differential_privacy" not in config:
            self.dp_params = None
        else:
            self.dp_params = config["differential_privacy"]
            print(f"Here are dp_params: {self.dp_params}")
            validate_dp_params(self.dp_params)
        self.global_model_tools = GlobalModelTools(
            global_model=self.global_model,
            example_model_state=self.model.state_dict(),
            collaborator_names=self.collaborator_names,
            dp_params=self.dp_params,
        )

    @aggregator
    def start(self):
        print("Performing initialization for model")
        self.collaborators = self.runtime.collaborators
        self.private = 10

        # determine starting round collaborators by grabbing first batch of loader
        round_collaborator_idxs = [
            batch[0] for batch in self.global_model_tools.global_data_loader
        ][0]
        self.round_collaborators = [
            self.collaborator_names[idx] for idx in round_collaborator_idxs
        ]

        if self.round is None:
            self.round = 0
            if self.dp_params is None:
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

        print("\n\n" + 20 * "#")
        print("Round {self.round}...")
        print("Training with collaborators: ", self.round_collaborators)
        print(20 * "#" + "\n\n")

        if (len(round_collaborator_idxs) != 0) and (
            round_collaborator_idxs[0].nelement() != 0
        ):
            self.next(
                self.aggregated_model_validation,
                foreach="round_collaborators",
                exclude=["private"],
            )
        else:
            if self.round == self.total_rounds - 1:
                print(
                    "Completed all rounds with no collaborator selected for training at any round"
                )
                self.next(self.end)

            self.round += 1
            self.next(self.start)

    @collaborator
    def aggregated_model_validation(self):
        print(f"Performing aggregated model validation for collaborator {self.input}")
        self.model = self.model.to(self.device)
        self.global_model = self.global_model.to(self.device)

        # verifying that model went to the correct GPU device
        assert next(self.model.parameters()).device == self.device
        assert next(self.global_model.parameters()).device == self.device

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
                global_model_state=self.global_model.state_dict(),
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
        self.model.load_state_dict(
            FedAvg(
                [input.model.cpu() for input in inputs],
                global_model_tools=self.global_model_tools,
                previous_global_state=self.global_model.cpu().state_dict(),
                dp_params=self.dp_params,
            ).state_dict()
        )

        if self.dp_params is not None:
            print(15 * "#")
            epsilon = self.global_model_tools.privacy_engine.get_epsilon(
                delta=self.dp_params["delta"]
            )
            print(
                "\nCurrent epsilon is: "
                + f"{epsilon} for delta: {self.dp_params['delta']}"
            )
            print(15 * "#")

        self.global_model.load_state_dict(deepcopy(self.model.state_dict()))
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
            if self.aggregated_model_accuracy > self.top_model_accuracy:
                print(
                    f"Accuracy improved to {self.aggregated_model_accuracy}"
                    + f"for round {self.round}"
                )
                self.top_model_accuracy = self.aggregated_model_accuracy
            self.round += 1

            # determine starting round collaborators by grabbing first batch of loader
            for batch in self.global_model_tools.global_data_loader:
                round_collaborator_idxs = batch
                break
            self.round_collaborators = [
                self.collaborator_names[idx] for idx in round_collaborator_idxs[0]
            ]

            print("\n\n" + 20 * "#")
            print(f"Round {self.round}...")
            print("Training with collaborators: ", self.round_collaborators)
            print(20 * "#" + "\n\n")

            if (len(round_collaborator_idxs) != 0) and (
                round_collaborator_idxs[0].nelement() != 0
            ):
                self.next(
                    self.aggregated_model_validation,
                    foreach="round_collaborators",
                    exclude=["private"],
                )
            else:
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
        "--config_path", help="Absolute path to the flow configuration file."
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

    # Setup collaborators with private attributes
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

    best_model = None
    initial_model = Net()
    top_model_accuracy = 0
    total_rounds = 10

    flflow = FederatedFlow(
        config_path=args.config_path,
        model=initial_model,
        collaborator_names=collaborator_names,
        device=device,
        total_rounds=total_rounds,
        top_model_accuracy=top_model_accuracy,
        clip_test=args.clip_test,
    )
    flflow.runtime = local_runtime
    flflow.run()
