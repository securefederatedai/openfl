"""Callbacks API."""


class Callback:
    def __init__(self):
        self._model = None
        self._tensor_db = None

    def set_model(self, model):
        self._model = model

    def set_tensor_db(self, tensor_db):
        self._tensor_db = tensor_db

    @property
    def tensor_db(self):
        return self._tensor_db

    @property
    def model(self):
        return self._model

    def on_task_begin(self, task: str, logs=None):
        """Callback function to be executed at the beginning of a task."""

    def on_task_end(self, task: str, logs=None):
        """Callback function to be executed at the end of a task."""

    def on_round_begin(self, round_num: int, logs=None):
        """Callback function to be executed at the beginning of a round."""

    def on_round_end(self, round_num: int, logs=None):
        """Callback function to be executed at the end of a round."""

    def on_experiment_begin(self, logs=None):
        """Callback function to be executed at the beginning of an experiment."""

    def on_experiment_end(self, logs=None):
        """Callback function to be executed at the end of an experiment."""
