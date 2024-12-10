from openfl.callbacks.callback import Callback


class LambdaCallback(Callback):
    """Custom on-the-fly callbacks.

    This callback can be constructed with functions that will be called
    at the appropriate time during the life-cycle of a Federated Learning experiment.
    Certain callbacks may expect positional arguments, for example:

    * on_round_begin: expects `round_num` as a positional argument.
    * on_round_end: expects `round_num` as a positional argument.

    Args:
        on_round_begin: called at the beginning of every round.
        on_round_end: called at the end of every round.
        on_experiment_begin: called at the beginning of an experiment.
        on_experiment_end: called at the end of an experiment.
        
    Example usage:

    In TaskRunner, you can define a list of callbacks for the aggregator or collaborator 
    in the plan, as follows:
    ```yaml
    aggregator :
        defaults : plan/defaults/aggregator.yaml
        template : openfl.component.Aggregator
        settings :
            ...
            callbacks : src.callbacks.aggregator_callbacks

    collaborator :
        defaults : plan/defaults/collaborator.yaml
        template : openfl.component.Collaborator
        settings :
            ...
            callbacks : src.callbacks.collaborator_callbacks
    ```

    where, `src.callbacks.aggregator_callbacks` and `src.callbacks.collaborator_callbacks` 
    are lists of callbacks as follows:

    ```python
    # src/callbacks.py
    def _print_memory_usage(round_num, logs):
        print(f"Round {round_num}, memory usage: {psutil.Process().memory_info().rss}")

    aggregator_callbacks = [
        LambdaCallback(on_round_begin=_print_memory_usage),
    ]

    collaborator_callbacks = [
        LambdaCallback(on_round_begin=_print_memory_usage),
    ]
    ```
    """
    def __init__(
        self,
        on_round_begin=None,
        on_round_end=None,
        on_experiment_begin=None,
        on_experiment_end=None,
    ):
        super().__init__()
        if on_round_begin is not None:
            self.on_round_begin = on_round_begin
        if on_round_end is not None:
            self.on_round_end = on_round_end
        if on_experiment_begin is not None:
            self.on_experiment_begin = on_experiment_begin
        if on_experiment_end is not None:
            self.on_experiment_end = on_experiment_end
