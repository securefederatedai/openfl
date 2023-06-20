from openfl.experimental.interface import FLSpec
from openfl.experimental.runtime import FederatedRuntime #, LocalRuntime
from openfl.experimental.placement import aggregator, collaborator
from openfl.experimental.interface import Aggregator, Collaborator

class TestFlow(FLSpec):
    def __init__(self, checkpoint: bool = False):
        super().__init__(checkpoint)

    @aggregator
    def start(self):
        self.collaborators = self.runtime.collaborators
        print("this is start of the flow")
        self.next(self.aggregator_step_1)

    @aggregator
    def aggregator_step_1(self):
        print("this is aggregator_step_1")
        self.next(self.aggregator_step_2)

    @aggregator
    def aggregator_step_2(self):
        print("this is aggregator_step_2")
        self.next(self.collaborator_step_1, foreach="collaborators")

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

        self.next(self.end)

    @aggregator
    def end(self):
        print("this is end of the flow")


def aggregator_private_attrs():
    return {
        "agg_pa": "Hi, I am aggregator"
    }

def collaborator_private_attrs(collab_name):
    return {
        "name": f"private attributes of {collab_name}"
    }
