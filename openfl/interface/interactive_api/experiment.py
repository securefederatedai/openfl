# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Python low-level API module."""
import functools
import os
from collections import defaultdict
from copy import deepcopy
from logging import getLogger
from pathlib import Path

from openfl.federated import Plan
from openfl.interface.cli import setup_logging
from openfl.interface.cli_helper import WORKSPACE
from openfl.utilities import split_tensor_dict_for_holdouts
from openfl.utilities.logs import write_metric


class FLExperiment:
    """Central class for FL experiment orchestration."""

    def __init__(self, federation, experiment_name='test', serializer_plugin=None) -> None:
        """
        Initialize an experiment inside a federation.

        Experiment makes sense in a scope of some machine learning problem.
        Information about the data on collaborators is contained on the federation level.
        """
        self.federation = federation
        self.experiment_name = experiment_name

        if serializer_plugin is None:
            self.serializer_plugin = \
                'openfl.plugins.interface_serializer.cloudpickle_serializer.CloudpickleSerializer'
        else:
            self.serializer_plugin = serializer_plugin

        self.logger = getLogger(__name__)
        setup_logging()

    def get_best_model(self):
        """Retrieve the model with the best score."""
        # Next line relies on aggregator inner field where model dicts are stored
        tensor_dict = self.federation.dir_client.get_best_model(
            experiment_name=self.experiment_name)
        self.task_runner_stub.rebuild_model(tensor_dict, validation=True, device='cpu')
        return self.task_runner_stub.model

    def get_last_model(self):
        """Retrieve the aggregated model after the last round."""
        # Next line relies on aggregator inner field where model dicts are stored
        tensor_dict = self.federation.dir_client.get_last_model(
            experiment_name=self.experiment_name)
        self.task_runner_stub.rebuild_model(tensor_dict, validation=True, device='cpu')
        return self.task_runner_stub.model

    def stream_metrics(self):
        """Stream metrics."""
        for metric_response in self.federation.dir_client.stream_metrics(self.experiment_name):
            print(
                metric_response.metric_origin,
                metric_response.task_name,
                metric_response.metric_name,
                float(metric_response.metric_value),
                metric_response.round)
            # print(
            write_metric(
                metric_response.metric_origin,
                metric_response.task_name,
                metric_response.metric_name,
                metric_response.metric_value,
                metric_response.round)

    def remove_experiment_data(self):
        """Remove experiment data."""
        log_message = 'Removing experiment data '
        if self.federation.dir_client.remove_experiment_data(
                experiment_name=self.experiment_name):
            log_message += 'succeed.'
        else:
            log_message += 'failed.'

        self.logger.info(log_message)

    def prepare_workspace_distribution(
            self, model_provider, task_keeper, data_loader,
            rounds_to_train,
            delta_updates=False, opt_treatment='RESET'):
        """Prepare an archive from a user workspace."""
        self._prepare_plan(model_provider, task_keeper, data_loader,
                           rounds_to_train,
                           delta_updates=delta_updates, opt_treatment=opt_treatment,
                           model_interface_file='model_obj.pkl',
                           tasks_interface_file='tasks_obj.pkl',
                           dataloader_interface_file='loader_obj.pkl')

        # Save serialized python objects to disc
        self._serialize_interface_objects(model_provider, task_keeper, data_loader)
        # Save the prepared plan
        Plan.dump(Path(f'./plan/{self.plan.name}'), self.plan.config, freeze=False)

        # PACK the WORKSPACE!
        # Prepare requirements file to restore python env
        self._export_python_env()

        # Compress te workspace to restore it on collaborator
        self.arch_path = self._pack_the_workspace()

        # DO CERTIFICATES exchange

    def start(self, *, model_provider, task_keeper, data_loader,
              rounds_to_train, delta_updates=False, opt_treatment='RESET'):
        """Prepare experiment and run."""
        self.prepare_workspace_distribution(
            model_provider, task_keeper, data_loader, rounds_to_train,
            delta_updates=delta_updates, opt_treatment=opt_treatment
        )
        self.logger.info('Starting experiment!')
        self.plan.resolve()
        initial_tensor_dict = self._get_initial_tensor_dict(model_provider)
        try:
            response = self.federation.dir_client.set_new_experiment(
                name=self.experiment_name,
                col_names=self.plan.authorized_cols,
                arch_path=self.arch_path,
                initial_tensor_dict=initial_tensor_dict
            )
        finally:
            # Remove the workspace archive
            os.remove(self.arch_path)
            del self.arch_path

        if response.accepted:
            self.logger.info('Experiment was accepted and launched.')
            self.logger.info('You can watch the experiment through tensorboard:')
            self.logger.info(response.tensorboard_address)
        else:
            self.logger.info('Experiment was not accepted or failed.')

    @staticmethod
    def _export_python_env():
        """Prepare requirements.txt."""
        from pip._internal.operations import freeze
        requirements_generator = freeze.freeze()
        with open('./requirements.txt', 'w') as f:
            for package in requirements_generator:
                # there is a pip bug with pkg-resources==0.0.0 req
                if '==' not in package or '0.0.0' in package:
                    # We do not export dependencies without version
                    continue
                f.write(package + '\n')

    @staticmethod
    def _pack_the_workspace():
        """Packing the archive."""
        from shutil import copytree
        from shutil import ignore_patterns
        from shutil import make_archive
        from shutil import rmtree
        from os import getcwd
        from os import makedirs
        from os.path import basename

        archive_type = 'zip'
        archive_name = basename(getcwd())

        tmp_dir = 'temp_' + archive_name
        makedirs(tmp_dir)

        ignore = ignore_patterns(
            '__pycache__', 'data', 'cert', tmp_dir, '*.crt', '*.key',
            '*.csr', '*.srl', '*.pem', '*.pbuf', '*zip')

        copytree('./', tmp_dir + '/workspace', ignore=ignore)

        arch_path = make_archive(archive_name, archive_type, tmp_dir + '/workspace')

        rmtree(tmp_dir)

        return arch_path

    def _get_initial_tensor_dict(self, model_provider):
        """Extract initial weights from the model."""
        self.task_runner_stub = self.plan.get_core_task_runner(model_provider=model_provider)
        tensor_dict, _ = split_tensor_dict_for_holdouts(
            self.logger,
            self.task_runner_stub.get_tensor_dict(False),
            **self.task_runner_stub.tensor_dict_split_fn_kwargs
        )
        return tensor_dict

    def _prepare_plan(self, model_provider, task_keeper, data_loader,
                      rounds_to_train,
                      delta_updates=False, opt_treatment='RESET',
                      model_interface_file='model_obj.pkl', tasks_interface_file='tasks_obj.pkl',
                      dataloader_interface_file='loader_obj.pkl'):
        """Fill plan.yaml file using provided setting."""
        # Create a folder to store plans
        os.makedirs('./plan', exist_ok=True)
        os.makedirs('./save', exist_ok=True)
        # Load the default plan
        base_plan_path = WORKSPACE / 'workspace/plan/plans/default/base_plan_interactive_api.yaml'
        plan = Plan.parse(base_plan_path, resolve=False)
        # Change plan name to default one
        plan.name = 'plan.yaml'

        # Seems like we still need to fill authorized_cols list
        # So aggregator know when to start sending tasks
        # We also could change the aggregator logic so it will send tasks to aggregator
        # as soon as it connects. This change should be a part of a bigger PR
        # brining in fault tolerance changes
        plan.authorized_cols = [shard_info.node_info.name for shard_info in
                                self.federation.get_shard_registry()]
        # Network part of the plan
        # We keep in mind that an aggregator FQND will be the same as the directors FQDN
        # We just choose a port randomly from plan hash
        plan.config['network']['settings']['agg_addr'] = \
            self.federation.director_node_fqdn.split(':')[0]  # We drop the port
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
                plan.config['tasks'][name] = {'function': name,
                                              'kwargs': task_keeper.task_settings[name]}
            else:
                # This is a validation type task (not altering the model state)
                for name_prefix, apply_kwarg in zip(['localy_tuned_model_', 'aggregated_model_'],
                                                    ['local', 'global']):
                    # We add two entries for this task: for local and global models
                    task_kwargs = deepcopy(task_keeper.task_settings[name])
                    task_kwargs.update({'apply': apply_kwarg})
                    plan.config['tasks'][name_prefix + name] = {
                        'function': name,
                        'kwargs': task_kwargs}

        # TaskRunner framework plugin
        # ['required_plugin_components'] should be already in the default plan with all the fields
        # filled with the default values
        plan.config['task_runner']['required_plugin_components'] = {}
        plan.config['task_runner']['required_plugin_components']['framework_adapters'] = \
            model_provider.framework_plugin

        # API layer
        plan.config['api_layer'] = {}
        plan.config['api_layer']['required_plugin_components'] = {}
        plan.config['api_layer']['settings'] = {}
        plan.config['api_layer']['required_plugin_components']['serializer_plugin'] = \
            self.serializer_plugin
        plan.config['api_layer']['settings'] = {
            'model_interface_file': model_interface_file,
            'tasks_interface_file': tasks_interface_file,
            'dataloader_interface_file': dataloader_interface_file, }

        plan.config['assigner']['settings']['task_groups'][0]['tasks'] = \
            [entry for entry in plan.config['tasks']
                if (type(plan.config['tasks'][entry]) is dict)
                and ('function' in plan.config['tasks'][entry])]
        self.plan = deepcopy(plan)

    def _serialize_interface_objects(self, model_provider, task_keeper, data_loader):
        """Save python objects to be restored on collaborators."""
        serializer = self.plan.build(
            self.plan.config['api_layer']['required_plugin_components']['serializer_plugin'], {})
        framework_adapter = Plan.build(model_provider.framework_plugin, {})
        # Model provider serialization may need preprocessing steps
        framework_adapter.serialization_setup()
        serializer.serialize(
            model_provider, self.plan.config['api_layer']['settings']['model_interface_file'])

        for object_, filename in zip(
                [task_keeper, data_loader],
                ['tasks_interface_file', 'dataloader_interface_file']):
            serializer.serialize(object_, self.plan.config['api_layer']['settings'][filename])


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

    def register_fl_task(self, model, data_loader, device, optimizer=None):
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
            contract = {'model': model, 'data_loader': data_loader,
                        'device': device, 'optimizer': optimizer}
            self.task_contract[training_method.__name__] = contract
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


class ModelInterface:
    """
    Registers model graph and optimizer.

    To be serialized and sent to collaborator nodes

    This is the place to determine correct framework adapter
        as they are needed to fill the model graph with trained tensors.

    There is no support for several models / optimizers yet.
    """

    def __init__(self, model, optimizer, framework_plugin) -> None:
        """
        Initialize model keeper.

        Tensors in provided graphs will be used for
        initialization of the global model.

        Arguments:
        model: Union[tuple, graph]
        optimizer: Union[tuple, optimizer]
        """
        self.model = model
        self.optimizer = optimizer
        self.framework_plugin = framework_plugin

    def provide_model(self):
        """Retrieve model."""
        return self.model

    def provide_optimizer(self):
        """Retrieve optimizer."""
        return self.optimizer


class DataInterface:
    """
    The class to define dataloaders.

    In the future users will have to adapt `unified data interface hook`
        in their dataloaders.
    For now, we can provide `data_path` variable on every collaborator node
        at initialization time for dataloader customization
    """

    def __init__(self, **kwargs):
        """Initialize DataLoader."""
        self.kwargs = kwargs

    @property
    def shard_descriptor(self):
        """Return shard descriptor."""
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor  will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor
        raise NotImplementedError

    def get_train_loader(self, **kwargs):
        """Output of this method will be provided to tasks with optimizer in contract."""
        raise NotImplementedError

    def get_valid_loader(self, **kwargs):
        """Output of this method will be provided to tasks without optimizer in contract."""
        raise NotImplementedError

    def get_train_data_size(self):
        """Information for aggregation."""
        raise NotImplementedError

    def get_valid_data_size(self):
        """Information for aggregation."""
        raise NotImplementedError
