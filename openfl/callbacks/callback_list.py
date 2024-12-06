from typing import List

from openfl.callbacks.callback import Callback


class CallbackList(Callback):
    def __init__(self, callbacks: List[Callback], model=None, tensor_db=None):
        super().__init__()
        self.callbacks = callbacks
        self.set_model(model)
        self.set_tensor_db(tensor_db)

    def set_model(self, model):
        if not model:
            return
        super().set_model(model)
        for callback in self.callbacks:
            callback.set_model(model)

    def set_tensor_db(self, tensor_db):
        if not tensor_db:
            return
        super().set_tensor_db(tensor_db)
        for callback in self.callbacks:
            callback.set_tensor_db(tensor_db)

    def on_task_begin(self, task: str, round_num: int, logs=None):
        for callback in self.callbacks:
            callback.on_task_begin(task, round_num, logs)

    def on_task_end(self, task: str, round_num: int, logs=None):
        for callback in self.callbacks:
            callback.on_task_end(task, round_num, logs)

    def on_round_begin(self, round_num: int, logs=None):
        for callback in self.callbacks:
            callback.on_round_begin(round_num, logs)

    def on_round_end(self, round_num: int, logs=None):
        for callback in self.callbacks:
            callback.on_round_end(round_num, logs)

    def on_experiment_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_experiment_begin(logs)

    def on_experiment_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_experiment_end(logs)
