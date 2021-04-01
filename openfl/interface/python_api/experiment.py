from typing import Union
import functools


class FLExperiment:
    def __init__(self, federation=None) -> None:
        self.federation = federation

    def start_experiment(self, model_provider, task_keeper, data_loader, rounds_to_train):
        # serializing objects
        # saving session
        # fixing requirements
        # packing the workspace

        # running tests

        # start an aggregator
        pass


class TaskInterface:
    """
    Task should accept the following entities that exist on collaborator nodes:
    1. model - will be rebuilt with relevant weights for every task by `TaskRunner`
    2. data_loader - data loader equipped with `repository adapter` that provides local data
    """
    def __init__(self) -> None:
        # Mapping 'task name' -> callable
        self.task_registry = dict()
        # Mapping 'task name' -> arguments
        self.task_contract = dict()
        # Mapping 'task name' -> arguments
        self.task_settings = dict()


    def register_fl_task(self, model, data_loader, device, optimizer=None):
        """
        This method is for registering FL tasks
        The task contract should be set up by providing:
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
        @TI.register_fl_task(model='my_model', data_loader='train_loader', device='device', optimizer='my_Adam_opt')
        def foo_task(my_model, train_loader, my_Adam_opt, device, batch_size, some_arg=356)
            ...
        `
        """
        # The highest level wrapper for allowing arguments for the decorator
        def decorator_with_args(training_method):
            # We could pass hooks to the decorator
            # @functools.wraps(training_method)
            def wrapper_decorator(**task_keywords):
                metric_dict = training_method(**task_keywords)
                return metric_dict

            # Saving the task and the contract for later serialization 
            self.task_registry[training_method.__name__] = wrapper_decorator
            contract = {'model':model, 'data_loader':data_loader, 'device':device, 'optimizer':optimizer}
            self.task_contract[training_method.__name__] = contract

            # We do not alter user environment
            return training_method

        return decorator_with_args


    def add_kwargs(self, **task_kwargs):
        """
        The method for registering tasks settings.
        This one is a decorator because we need task name and
        to be consistent with the main registering method
        """
        # The highest level wrapper for allowing arguments for the decorator
        def decorator_with_args(training_method):
            # Saving the task's settings to be written in plan 
            self.task_settings[training_method.__name__] = task_kwargs

            return training_method

        return decorator_with_args


class ModelInterface:
    '''
    Registers model graph (s) and optimizer (s)
        to be serialized and sent to collaborator nodes

    This is the place to determine correct framework adapter
        as they are needed to fill the model graph with trained tensors
    '''
    def __init__(self, model, optimizer, framework_plugin) -> None:
        '''
        Arguments:
        model: Union[tuple, graph]
        optimizer: Union[tuple, optimizer]

        Caution! 
        Tensors in provided graphs will be used for
        initialization of the global model.
        '''
        self.model = model
        self.optimizer = optimizer
        self.framework_plugin = framework_plugin
        
        
    def provide_model(self):
        return self.model

    def provide_optimizer(self):
        return self.optimizer


class DataInterface:
    """
    The class to define dataloaders
    In the future users will have to adapt `unified data interface hook`
        in their dataloaders.
    For now, we can provide `data_path` variable on every collaborator node
        at initialization time for dataloader customization
    """
    def set_data_path(self, data_path) -> None:
        self.data_path = data_path


    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        raise NotImplementedError

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        raise NotImplementedError

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        raise NotImplementedError

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        raise NotImplementedError

# class test(DataInterface, ModelInterface):
#     def __init__(self, data_path) -> None:
#         super().__init__(data_path)