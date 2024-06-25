# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.interface import FLSpec
from openfl.experimental.placement import aggregator, collaborator


class VerticalFlow(FLSpec):

    def __init__(self, checkpoint: bool = False):
        super().__init__(checkpoint)

    @aggregator
    def start(self):
        self.collaborators = self.runtime.collaborators
        self.round = 0
        self.next_collaborator = ['portland']
        self.next(self.custom_task_portland, foreach='next_collaborator')

    @collaborator
    def custom_task_portland(self):
        print(f'Collaborator {self.input}: performing custom task')
        self.result = 0
        self.next(self.gather_portland_results)

    @aggregator
    def gather_portland_results(self, inputs):
        self.results = []
        self.results.append(inputs[0].result)
        self.next_collaborator = ['seattle']
        self.next(self.custom_task_seattle, foreach='next_collaborator', exclude=['results'])

    @collaborator
    def custom_task_seattle(self):
        print(f'Collaborator {self.input}: performing custom task')
        self.result = 1
        self.next(self.gather_seattle_results)

    @aggregator
    def gather_seattle_results(self, inputs):
        self.results.append(inputs[0].result)
        self.next_collaborator = ['chandler']
        self.next(self.custom_task_chandler, foreach='next_collaborator', exclude=['results'])

    @collaborator
    def custom_task_chandler(self):
        print(f'Collaborator {self.input}: performing custom task')
        self.result = 2
        self.next(self.gather_chandler_results)

    @aggregator
    def gather_chandler_results(self, inputs):
        self.results.append(inputs[0].result)
        self.next_collaborator = ['bangalore']
        self.next(self.custom_task_bangalore, foreach='next_collaborator', exclude=['results'])

    @collaborator
    def custom_task_bangalore(self):
        print(f'Collaborator {self.input}: performing custom task')
        self.result = 3
        self.next(self.gather_bangalore_results)

    @aggregator
    def gather_bangalore_results(self, inputs):
        self.results.append(inputs[0].result)
        self.next(self.combine)

    @aggregator
    def combine(self):
        print(f'The results from each of the collaborators are: {self.results}')
        print(f'Their average = {sum(self.results) / len(self.results)}')
        self.round += 1
        if self.round < 10:
            print()
            print(f'Starting round {self.round}...')
            self.next_collaborator = ['portland']
            self.next(self.custom_task_portland, foreach='next_collaborator')
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print('This is the end of the flow')
