# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openfl.callbacks.callback import Callback
from openfl.callbacks.memory_profiler import MemoryProfiler


class CallbackList(Callback):
    """An ensemble of callbacks.

    This class allows multiple callbacks to be used together, by sequentially
    calling each callback's respective methods.

    Attributes:
        callbacks: A list of `openfl.callbacks.Callback` instances.
        add_memory_profiler: If True, adds a `MemoryProfiler` callback to the list.
        tensor_db: Optional `TensorDB` instance of the respective participant.
            If provided, callbacks can access TensorDB for various actions.
        params: Additional parameters saved for use within the callbacks.
    """

    def __init__(
        self,
        callbacks: list,
        add_memory_profiler=False,
        tensor_db=None,
        **params,
    ):
        super().__init__()
        self.callbacks = _flatten(callbacks) if callbacks else []

        self._add_default_callbacks(add_memory_profiler)

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

    def _add_default_callbacks(self, add_memory_profiler):
        self._memory_profiler = None
        for cb in self.callbacks:
            if isinstance(cb, MemoryProfiler):
                self._memory_profiler = cb

        if add_memory_profiler and self._memory_profiler is None:
            self._memory_profiler = MemoryProfiler()
            self.callbacks.append(self._memory_profiler)

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


def _flatten(l):
    """Flatten a possibly-nested tree of lists."""
    if not isinstance(l, (list, tuple)):
        return [l]
    for elem in l:
        if isinstance(elem, list):
            yield from _flatten(elem)
        else:
            yield elem
