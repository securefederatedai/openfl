from typing import List

from openfl.callbacks.callback import Callback


class CallbackList(Callback):
    """An ensemble of callbacks.
    
    This class allows multiple callbacks to be used together, by sequentially
    calling each callback's respective methods.

    Attributes:
        callbacks: A list of `openfl.callbacks.Callback` instances.
        tensor_db: Optional `TensorDB` instance of the respective participant.
            If provided, callbacks can access TensorDB for various actions.
        params: Additional parameters saved for use within the callbacks.
    """
    def __init__(self, callbacks: List[Callback], tensor_db=None, **params):
        super().__init__()
        self.callbacks = callbacks
        self.set_tensor_db(tensor_db)
        self.set_params(params)

    def set_params(self, params):
        self.params = params
        if params:
            for callback in self.callbacks:
                callback.set_params(params)

    def set_tensor_db(self, tensor_db):
        self.tensor_db = tensor_db
        if tensor_db:
            for callback in self.callbacks:
                callback.set_tensor_db(tensor_db)

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
