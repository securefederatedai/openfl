# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.interface import FLSpec
from openfl.experimental.placement import aggregator, collaborator
from torch import nn, optim


class VerticalTwoPartyFlow(FLSpec):

    def __init__(self, batch_num=0, checkpoint: bool = False):
        super().__init__(checkpoint)
        self.batch_num = batch_num

    @aggregator
    def start(self):
        self.collaborators = self.runtime.collaborators
        print(f'Batch_num = {self.batch_num}')
        # 1) Zero the gradients
        self.label_model_optimizer.zero_grad()
        self.next(self.data_model_forward_pass, foreach='collaborators')

    @collaborator
    def data_model_forward_pass(self):
        self.data_model_output_local = ''
        for idx, (images, _) in enumerate(self.trainloader):
            if idx < self.batch_num:
                continue
            self.data_model_optimizer.zero_grad()
            images = images.view(images.shape[0], -1)
            model_output = self.data_model(images)
            self.data_model_output_local = model_output
            self.data_model_output = model_output.detach().requires_grad_()
            break
        self.next(self.label_model_forward_pass)

    @aggregator
    def label_model_forward_pass(self, inputs):
        criterion = nn.NLLLoss()
        self.grad_to_local = []
        total_loss = 0
        self.data_remaining = False
        for idx, (_, labels) in enumerate(self.trainloader):
            if idx < self.batch_num:
                continue
            self.data_remaining = True
            pred = self.label_model(inputs[0].data_model_output)
            loss = criterion(pred, labels)
            loss.backward()
            self.grad_to_local = inputs[0].data_model_output.grad.clone()
            self.label_model_optimizer.step()
            total_loss += loss
            break
        print(f'Total loss = {total_loss}')
        self.next(self.data_model_backprop, foreach='collaborators')

    @collaborator
    def data_model_backprop(self):
        if self.data_remaining:
            self.data_model_optimizer = optim.SGD(self.data_model.parameters(), lr=0.03)
            self.data_model_optimizer.zero_grad()
            self.data_model_output_local.backward(self.grad_to_local)
            self.data_model_optimizer.step()
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        print(f'Join batch_num = {self.batch_num}')
        self.batch_num += 1
        self.next(self.end)

    @aggregator
    def end(self):
        print('This is the end of the flow')
