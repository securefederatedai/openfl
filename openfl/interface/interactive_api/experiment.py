# We need assigner interface
import os
from copy import deepcopy
from logging import getLogger
import functools
from collections import defaultdict
from openfl import federated
from openfl.federated import Plan
from pathlib import Path
from importlib import resources

from openfl.utilities import split_tensor_dict_for_holdouts

class FLExperiment:
    def __init__(self, federation, serializer_plugin=None) -> None:
        self.federation = federation

        if serializer_plugin is None:
            self.serializer_plugin = 'openfl.plugins.interface_serializer.cloudpickle_serializer.Cloudpickle_Serializer'
        else:
            self.serializer_plugin = serializer_plugin

        self.logger = getLogger(__name__)

    def get_best_model(self):
        # Next line relies on aggregator inner field where model dicts are stored
        best_tensor_dict = self.server.aggregator.best_tensor_dict
        self.task_runner_stub.rebuild_model(best_tensor_dict, validation=True, device='cpu')
        return self.task_runner_stub.model

    def prepare_workspace_distribution(self, model_provider, task_keeper, data_loader, \
            rounds_to_train, \
            delta_updates=False, opt_treatment='RESET'):

        self._prepare_plan(model_provider, task_keeper, data_loader, \
            rounds_to_train, \
            delta_updates=delta_updates, opt_treatment=opt_treatment, \
            model_interface_file='model_obj.pkl', tasks_interface_file='tasks_obj.pkl', \
                                            dataloader_interface_file='loader_obj.pkl')

        # Save serialized python objects to disc
        self._serialize_interface_objects(model_provider, task_keeper, data_loader)
        # Save the prepared plan
        Plan.Dump(Path('./plan/plan.yaml'), self.plan.config, freeze=False)

        # PACK the WORKSPACE!
        # Prepare requirements file to restore python env
        self._export_python_env()

        # Compress te workspace to restore it on collaborator
        self._pack_the_workspace()

        # DO CERTIFICATES exchange

    def start_experiment(self, model_provider):
        '''
        This method also requires model_provider to start experiment with another initialization
        without workspace redistribution.
        '''
        # Start the aggregator
        self.plan.resolve()

        initial_tensor_dict = self._get_initial_tensor_dict(model_provider)
        self.server = self.plan.interactive_api_get_server(initial_tensor_dict,
            chain=self.federation.cert_chain, certificate=self.federation.agg_certificate,
            private_key=self.federation.agg_private_key)

        self.server.serve()
        # return server

    @staticmethod
    def _export_python_env():
        from pip._internal.operations import freeze
        requirements_generator = freeze.freeze()
        with open('./requirements.txt', 'w') as f:
            for package in requirements_generator:
                if 'openfl' not in package:
                    f.write(package + '\n')


    @staticmethod
    def _pack_the_workspace():
        from shutil import make_archive, copytree, ignore_patterns, rmtree
        from os import getcwd, makedirs
        from os.path import basename


        archiveType = 'zip'
        archiveName = basename(getcwd())
        # archiveFileName = archiveName + '.' + archiveType

        tmpDir = 'temp_' + archiveName
        makedirs(tmpDir)

        ignore = ignore_patterns(
            '__pycache__', 'data', 'cert', tmpDir, '*.crt', '*.key', '*.csr', '*.srl', '*.pem', '*.pbuf')

        copytree('./', tmpDir + '/workspace', ignore=ignore) 

        make_archive(archiveName, archiveType, tmpDir + '/workspace')

        rmtree(tmpDir)


    def _get_initial_tensor_dict(self, model_provider):
        self.task_runner_stub = self.plan.get_core_task_runner(model_provider=model_provider)
        tensor_dict, _ = split_tensor_dict_for_holdouts(
            self.logger,
            self.task_runner_stub.get_tensor_dict(False),
            **self.task_runner_stub.tensor_dict_split_fn_kwargs
        )
        return tensor_dict


    def _prepare_plan(self, model_provider, task_keeper, data_loader, \
            rounds_to_train, \
            delta_updates=False, opt_treatment='RESET', \
            model_interface_file='model_obj.pkl', tasks_interface_file='tasks_obj.pkl', \
                                            dataloader_interface_file='loader_obj.pkl'):
        # Load the default plan
        with resources.path('openfl.interface.interactive_api', 'plan.yaml') as plan_path:
            plan = Plan.Parse(Path(plan_path), resolve=False)

        plan.authorized_cols = list(self.federation.col_data_paths.keys())
        # Network part of the plan
        plan.config['network']['settings']['agg_addr'] = self.federation.fqdn
        plan.config['network']['settings']['disable_tls'] = self.federation.disable_tls

        # Aggregator part of the plan
        plan.config['aggregator']['settings']['rounds_to_train'] = rounds_to_train

        # Collaborator part
        plan.config['collaborator']['settings']['delta_updates'] = delta_updates
        plan.config['collaborator']['settings']['opt_treatment'] = opt_treatment

        # DataLoader part
        for setting, value in data_loader.kwargs.items():
             plan.config['data_loader']['settings'][setting] = value

        # Tasks part
        for name in task_keeper.task_registry:
            if task_keeper.task_contract[name]['optimizer'] is not None:
                # This is training task
                plan.config['tasks'][name] = {'function':name,
                                            'kwargs':task_keeper.task_settings[name]}
            else:
                # This is a validation type task (not altering the model state)
                for name_prefix, apply_kwarg in zip(['localy_tuned_model_', 'aggregated_model_'],
                                                ['local', 'global']):
                    # We add two entries for this task: for local and global models
                    task_kwargs = deepcopy(task_keeper.task_settings[name])
                    task_kwargs.update({'apply':apply_kwarg})
                    plan.config['tasks'][name_prefix + name] = {'function':name,
                                    'kwargs':task_kwargs}


        # TaskRunner framework plugin
        # ['required_plugin_components'] should be already in the default plan with all the fields filled
        # with the default values
        plan.config['task_runner']['required_plugin_components'] = dict()
        plan.config['task_runner']['required_plugin_components']['framework_adapters'] = model_provider.framework_plugin


        # API layer
        plan.config['api_layer'] = dict()
        plan.config['api_layer']['required_plugin_components'] = dict()
        plan.config['api_layer']['settings'] = dict()
        plan.config['api_layer']['required_plugin_components']['serializer_plugin'] = \
                self.serializer_plugin
        plan.config['api_layer']['settings'] = {'model_interface_file' : model_interface_file,
                                                'tasks_interface_file' : tasks_interface_file,
                                                'dataloader_interface_file' : dataloader_interface_file,}


        plan.config['assigner']['settings']['task_groups'][0]['tasks'] = \
            [entry for entry in plan.config['tasks'] if (type(plan.config['tasks'][entry]) is dict) \
                and ('function' in plan.config['tasks'][entry])]
        self.plan = deepcopy(plan)
        os.makedirs('./plan', exist_ok=True)
        os.makedirs('./save', exist_ok=True)
        

    def _serialize_interface_objects(self, model_provider, task_keeper, data_loader):
        serializer = self.plan.Build(self.plan.config['api_layer']['required_plugin_components']['serializer_plugin'], {})
        framework_adapter = Plan.Build(model_provider.framework_plugin, {})
        # Model provider serialization may need preprocessing steps
        framework_adapter.serialization_setup()
        serializer.serialize(model_provider, self.plan.config['api_layer']['settings']['model_interface_file'])

        for object_, filename in zip([task_keeper, data_loader],
                    ['tasks_interface_file', 'dataloader_interface_file']):
            serializer.serialize(object_, self.plan.config['api_layer']['settings'][filename])



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
        self.task_settings = defaultdict(dict)


    def register_fl_task(self, model, data_loader, device, optimizer=None):
        """
        This method is for registering FL tasks
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
        @TI.register_fl_task(model='my_model', data_loader='train_loader', device='device', optimizer='my_Adam_opt')
        def foo_task(my_model, train_loader, my_Adam_opt, device, batch_size, some_arg=356)
            ...
        `
        """
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
    def _delayed_init(self, data_path):
        raise NotImplementedError


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
