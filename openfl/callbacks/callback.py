# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class Callback:
    """Base class for callbacks.

    Callbacks can be used to perform actions at different stages of the
    Federated Learning process. To create a custom callback, subclass
    `openfl.callbacks.Callback` and implement the necessary methods.

    Callbacks can be triggered on the aggregator and collaborator side
    for the following events:
        * At the beginning of an experiment
        * At the beginning of a round
        * At the end of a round
        * At the end of an experiment

    Attributes:
        params: Additional parameters saved for use within the callback.
        tensor_db: The `TensorDB` instance of the respective participant.
    """

    def __init__(self):
        self.params = None
        self.tensor_db = None

    def set_params(self, params):
        self.params = params

    def set_tensor_db(self, tensor_db):
        self.tensor_db = tensor_db

    def on_round_begin(self, round_num: int, logs=None):
        """Callback function to be executed at the beginning of a round.

        Subclasses need to implement actions to be taken here.
        """

    def on_round_end(self, round_num: int, logs=None):
        """Callback function to be executed at the end of a round.

        Subclasses need to implement actions to be taken here.
        """

    def on_experiment_begin(self, logs=None):
        """Callback function to be executed at the beginning of an experiment.

        Subclasses need to implement actions to be taken here.
        """

    def on_experiment_end(self, logs=None):
        """Callback function to be executed at the end of an experiment.

        Subclasses need to implement actions to be taken here.
        """

    def on_task_begin(
        self,
        round_num: int,
        logs=None,
    ):
        """Callback function to be executed at the beginning of a task.

        Subclasses need to implement actions to be taken here.
        """

    def on_task_end(self, round_num: int, logs=None):
        """Callback function to be executed at the end of a task.

        Subclasses need to implement actions to be taken here.
        """
