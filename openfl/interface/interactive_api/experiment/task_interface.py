import functools
from collections import defaultdict

from openfl.component.aggregation_functions import AggregationFunctionInterface
from openfl.component.aggregation_functions import WeightedAverage


class TaskInterface:
    """
    Task keeper class.

    Task should accept the following entities that exist on collaborator nodes:
    1. model - will be rebuilt with relevant weights for every task by `TaskRunner`
    2. data_loader - data loader equipped with `repository adapter` that provides local data
    3. device - a device to be used on collaborator machines
    4. optimizer (optional)

    Task returns a dictionary {metric name: metric value for this task}
    """

    def __init__(self) -> None:
        """Initialize task registry."""
        # Mapping 'task name' -> callable
        self.task_registry = {}
        # Mapping 'task name' -> arguments
        self.task_contract = {}
        # Mapping 'task name' -> arguments
        self.task_settings = defaultdict(dict)
        # Mapping task name -> callable
        self.aggregation_types = {}

    def register_fl_task(self, model, data_loader, device, optimizer=None, aggregation_type=None):
        """
        Register FL tasks.

        The task contract should be set up by providing variable names:
        [model, data_loader, device] - necessarily
        and optimizer - optionally

        All tasks should accept contract entities to be run on collaborator node.
        Moreover we ask users return dict{'metric':value} in every task
        `
        TI = TaskInterface()

        task_settings = {
            'batch_size': 32,
            'some_arg': 228,
        }
        @TI.add_kwargs(**task_settings)
        @TI.register_fl_task(model='my_model', data_loader='train_loader',
                device='device', optimizer='my_Adam_opt')
        def foo_task(my_model, train_loader, my_Adam_opt, device, batch_size, some_arg=356)
            ...
        `
        """
        if aggregation_type is None:
            aggregation_type = WeightedAverage()

        # The highest level wrapper for allowing arguments for the decorator
        def decorator_with_args(training_method):
            # We could pass hooks to the decorator
            # @functools.wraps(training_method)
            functools.wraps(training_method)

            def wrapper_decorator(**task_keywords):
                metric_dict = training_method(**task_keywords)
                return metric_dict

            # Saving the task and the contract for later serialization
            self.task_registry[training_method.__name__] = wrapper_decorator
            contract = {
                'model': model,
                'data_loader': data_loader,
                'device': device,
                'optimizer': optimizer
            }
            self.task_contract[training_method.__name__] = contract
            self.aggregation_types[training_method.__name__] = aggregation_type
            # We do not alter user environment
            return training_method

        return decorator_with_args

    def add_kwargs(self, **task_kwargs):
        """
        Register tasks settings.

        Warning! We do not actually need to register additional kwargs,
        we ust serialize them.
        This one is a decorator because we need task name and
        to be consistent with the main registering method
        """
        # The highest level wrapper for allowing arguments for the decorator
        def decorator_with_args(training_method):
            # Saving the task's settings to be written in plan
            self.task_settings[training_method.__name__] = task_kwargs

            return training_method

        return decorator_with_args
