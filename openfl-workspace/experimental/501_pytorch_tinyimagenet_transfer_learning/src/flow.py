# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
from itertools import repeat
from typing import Callable, List, Optional, Union, Tuple, Sequence, Any
from types import FunctionType

import numpy as np
import tqdm
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from openfl.experimental.interface import FLSpec
from openfl.experimental.placement import aggregator, collaborator


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _log_api_usage_once(obj: Any) -> None:
    module = obj.__module__
    if not module.startswith('torchvision'):
        module = f'torchvision.internal.{module}'
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f'{module}.{name}')


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = (len(kernel_size)
                             if isinstance(kernel_size, Sequence)
                             else len(dilation))
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use"
                + " Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=nn.ReLU6
                                     )
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        classifier_block: Optional = None
    ) -> None:
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                + f"or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(
            input_channel * width_mult, round_nearest
        )
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        features: List[nn.Module] = [
            Conv2dNormActivation(
                3, input_channel, stride=2, norm_layer=norm_layer,
                activation_layer=nn.ReLU6
            )
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel, output_channel, stride,
                        expand_ratio=t, norm_layer=norm_layer
                    )
                )
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1,
                norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = classifier_block

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


classifier_block = nn.Sequential
dropout = nn.Dropout
linear_layer = nn.Linear
in_features = _make_divisible(1280 * 1.0, 8)
inverted_residual_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class Net(nn.Module):
    def __init__(self, mobilenetv2, in_features, out_features):
        super(Net, self).__init__()
        self.base_model = mobilenetv2
        self.base_model.requires_grad_(False)
        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )
        self.linear.requires_grad_(True)

    def forward(self, x):
        x = self.base_model.forward(x)
        x = self.linear(x)
        return x


def inference(network, test_loader):
    network = network.to(device).eval()

    data_loader = tqdm.tqdm(test_loader, desc="validate")
    val_score = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in data_loader:
            samples = target.shape[0]
            total_samples += samples
            data, target = torch.tensor(data).to(device), torch.tensor(
                target).to(device, dtype=torch.int64)
            output = network(data)
            pred = output.argmax(dim=1, keepdim=True)
            val_score += pred.eq(target).sum().cpu().numpy()

    accuracy = (val_score / total_samples)
    print(f'Validation Accuracy: {100*accuracy:.2f}%')
    return accuracy


def fedavg(models):
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        state_dict[key] = np.sum(
            np.array([state[key] for state in state_dicts], dtype=object), axis=0
        ) / len(models)
    new_model.load_state_dict(state_dict)
    return new_model


class TinyImageNetFlow(FLSpec):
    def __init__(self, model=None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.rounds = rounds

    @aggregator
    def start(self):
        print('Performing initialization for model')
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.current_round = 0
        self.model.base_model.load_state_dict(
            self.pretrained_state_dict
        )
        self.next(
            self.aggregated_model_validation,
            foreach="collaborators",
            exclude=["private"],
        )

    @collaborator
    def aggregated_model_validation(self):
        print(f"Performing aggregated model validation for collaborator {self.input}")
        self.agg_validation_score = inference(self.model, self.test_loader)
        print(f"{self.input} value of {self.agg_validation_score}")
        self.next(self.train)

    @collaborator
    def train(self):
        data_loader = tqdm.tqdm(self.train_loader, desc="train")
        self.model = self.model.to(device).train()

        losses = []
        self.optimizer = optim.Adam(
            [x for x in self.model.parameters() if x.requires_grad],
            lr=1e-4
        )
        for data, target in data_loader:
            data, target = torch.tensor(data).to(device), torch.tensor(
                target).to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().numpy())
            self.loss = loss.item()

        print(f'train_loss: {np.mean(losses)}')
        self.training_completed = True
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = inference(self.model, self.test_loader)
        print(
            "Doing local model validation"
            + f"for collaborator {self.input}: {self.local_validation_score}"
        )
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
            "Average aggregated model "
            + f"validation values = {self.aggregated_model_accuracy}"
        )
        print(f"Average training loss = {self.average_loss}")
        print(f"Average local model validation values = {self.local_model_accuracy}")
        self.model = fedavg([input.model.to("cpu") for input in inputs])
        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        self.current_round += 1
        if self.current_round < self.rounds:
            self.next(
                self.aggregated_model_validation, foreach="collaborators", exclude=["private"])
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print("This is the end of the flow")
