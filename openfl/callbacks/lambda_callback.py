from openfl.callbacks.callback import Callback


class LambdaCallback(Callback):
    def __init__(
        self,
        on_task_begin=None,
        on_task_end=None,
        on_round_begin=None,
        on_round_end=None,
        on_experiment_begin=None,
        on_experiment_end=None,
    ):
        super().__init__()
        if on_task_begin is not None:
            self.on_task_begin = on_task_begin
        if on_task_end is not None:
            self.on_task_end = on_task_end
        if on_round_begin is not None:
            self.on_round_begin = on_round_begin
        if on_round_end is not None:
            self.on_round_end = on_round_end
        if on_experiment_begin is not None:
            self.on_experiment_begin = on_experiment_begin
        if on_experiment_end is not None:
            self.on_experiment_end = on_experiment_end
