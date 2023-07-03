from openfl.experimental.interface import FLSpec
from openfl.experimental.placement import aggregator, collaborator


class TestFlow(FLSpec):
    def __init__(self, rounds: int = 10, checkpoint: bool = False):
        super().__init__(checkpoint)
        self.total_round_number = rounds
        self.round_number = 0

    @aggregator
    def start(self):
        self.collaborators = self.runtime.collaborators
        print("this is start of the flow")
        self.start_attr = "hi, I am still present"
        self.next(self.aggregator_step_1)

    @aggregator
    def aggregator_step_1(self):
        print("this is aggregator_step_1")
        print(self.start_attr)
        print(f"self.round_number: {self.round_number}")
        self.next(self.aggregator_step_2)

    @aggregator
    def aggregator_step_2(self):
        print("this is aggregator_step_2")
        self.next(self.collaborator_step_1, foreach="collaborators", exclude=["start_attr",])

    @collaborator
    def collaborator_step_1(self):
        print(f"this is collaborator 1, my name is: {self.input}")
        print(f"my private attributes are: {self.name}")
        self.next(self.collaborator_step_2)

    @collaborator
    def collaborator_step_2(self):
        print(f"this is collaborator 2, my name is: {self.input}")
        print(f"my private attributes are: {self.name}")
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        print("this is join step")
        print("i have access to workspace of all collaborators, but not private attrs")
        for input_ in inputs:
            print(f"Hi, my name is {input_.input}")
        print(self.start_attr)
        self.next(self.collaborator_step_3, foreach="collaborators")

    @collaborator
    def collaborator_step_3(self):
        print(f"this is collaborator 3, my name is: {self.input}")
        print(f"my private attributes are: {self.name}")
        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        if self.round_number < self.total_round_number:
            self.round_number += 1
            self.next(self.aggregator_step_1)
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print("this is end of the flow")
