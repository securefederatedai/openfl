"""Callbacks API."""


class Callback:
    def __init__(self):
        self._model = None

    def set_model(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    def on_task_begin(self, task: str, round_num: int, logs=None):
        """Callback function to be executed at the beginning of a task."""

    def on_task_end(self, task: str, round_num: int, logs=None):
        """Callback function to be executed at the end of a task."""

    def on_round_begin(self, round_num: int, logs=None):
        """Callback function to be executed at the beginning of a round."""

    def on_round_end(self, round_num: int, logs=None):
        """Callback function to be executed at the end of a round."""

    def on_experiment_begin(self, logs=None):
        """Callback function to be executed at the beginning of an experiment."""

    def on_experiment_end(self, logs=None):
        """Callback function to be executed at the end of an experiment."""
