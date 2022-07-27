# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Python low-level API module."""
import os
import time
from collections import defaultdict
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Dict
from typing import Tuple

from tensorboardX import SummaryWriter

from openfl.interface.aggregation_functions import AggregationFunction
from openfl.interface.aggregation_functions import WeightedAverage
from openfl.component.assigner.tasks import Task
from openfl.component.assigner.tasks import TrainTask
from openfl.component.assigner.tasks import ValidateTask
from openfl.federated import Plan
from openfl.interface.cli import setup_logging
from openfl.interface.cli_helper import WORKSPACE
from openfl.native import update_plan
from openfl.utilities import split_tensor_dict_for_holdouts
from openfl.utilities.workspace import dump_requirements_file


class ModelStatus:
    """Model statuses."""

    INITIAL = 'initial'
    BEST = 'best'
    LAST = 'last'
    RESTORED = 'restored'


class FLExperiment:
    """Central class for FL experiment orchestration."""

    def __init__(
            self,
            federation,
            experiment_name: str = None,
            serializer_plugin: str = 'openfl.plugins.interface_serializer.'
                                     'cloudpickle_serializer.CloudpickleSerializer'
    ) -> None:
        """
        Initialize an experiment inside a federation.

        Experiment makes sense in a scope of some machine learning problem.
        Information about the data on collaborators is contained on the federation level.
        """
        self.federation = federation
        self.experiment_name = experiment_name or 'test-' + time.strftime('%Y%m%d-%H%M%S')
        self.summary_writer = None
        self.serializer_plugin = serializer_plugin

        self.experiment_accepted = False

        self.is_validate_task_exist = False

        self.logger = getLogger(__name__)
        setup_logging()

        self._initialize_plan()


    def _initialize_plan(self):
        """Setup plan from base plan interactive api."""
        # Create a folder to store plans
        os.makedirs('./plan', exist_ok=True)
        os.makedirs('./save', exist_ok=True)
        # Load the default plan
        base_plan_path = WORKSPACE / 'workspace/plan/plans/default/base_plan_interactive_api.yaml'
        plan = Plan.parse(base_plan_path, resolve=False)
        # Change plan name to default one
        plan.name = 'plan.yaml'

        self.plan = deepcopy(plan)
    
    def _assert_experiment_accepted(self):
        """Assure experiment is sent to director."""
        if not self.experiment_accepted:
            self.logger.error('The experiment has not been accepted by director')
            self.logger.error(
                'Report the experiment first: '
                'use the Experiment.start() method.')
            raise Exception

    def get_best_model(self):
        """Retrieve the model with the best score."""
        self._assert_experiment_accepted()
        tensor_dict = self.federation.dir_client.get_best_model(
            experiment_name=self.experiment_name)

        return self._rebuild_model(tensor_dict, upcoming_model_status=ModelStatus.BEST)

    def get_last_model(self):
        """Retrieve the aggregated model after the last round."""
        self._assert_experiment_accepted()
        tensor_dict = self.federation.dir_client.get_last_model(
            experiment_name=self.experiment_name)

        return self._rebuild_model(tensor_dict, upcoming_model_status=ModelStatus.LAST)

    def _rebuild_model(self, tensor_dict, upcoming_model_status=ModelStatus.BEST):
        """Use tensor dict to update model weights."""
        if len(tensor_dict) == 0:
            warning_msg = ('No tensors received from director\n'
                           'Possible reasons:\n'
                           '\t1. Aggregated model is not ready\n'
                           '\t2. Experiment data removed from director')

            if upcoming_model_status == ModelStatus.BEST and not self.is_validate_task_exist:
                warning_msg += '\n\t3. No validation tasks are provided'

            warning_msg += f'\nReturn {self.current_model_status} model'

            self.logger.warning(warning_msg)

        else:
            self.task_runner_stub.rebuild_model(tensor_dict, validation=True, device='cpu')
            self.current_model_status = upcoming_model_status

        return deepcopy(self.task_runner_stub.model)

    def stream_metrics(self, tensorboard_logs: bool = True) -> None:
        """Stream metrics."""
        self._assert_experiment_accepted()
        for metric_message_dict in self.federation.dir_client.stream_metrics(self.experiment_name):
            self.logger.metric(
                f'Round {metric_message_dict["round"]}, '
                f'collaborator {metric_message_dict["metric_origin"]} '
                f'{metric_message_dict["task_name"]} result '
                f'{metric_message_dict["metric_name"]}:\t{metric_message_dict["metric_value"]:f}')

            if tensorboard_logs:
                self.write_tensorboard_metric(metric_message_dict)

    def write_tensorboard_metric(self, metric: dict) -> None:
        """Write metric callback."""
        if not self.summary_writer:
            self.summary_writer = SummaryWriter(f'./logs/{self.experiment_name}', flush_secs=5)

        self.summary_writer.add_scalar(
            f'{metric["metric_origin"]}/{metric["task_name"]}/{metric["metric_name"]}',
            metric['metric_value'], metric['round'])

    def remove_experiment_data(self):
        """Remove experiment data."""
        self._assert_experiment_accepted()
        log_message = 'Removing experiment data '
        if self.federation.dir_client.remove_experiment_data(
                name=self.experiment_name
        ):
            log_message += 'succeed.'
            self.experiment_accepted = False
        else:
            log_message += 'failed.'

        self.logger.info(log_message)

    def prepare_workspace_distribution(self, model_provider, task_keeper, data_loader,
                                       task_assigner,
                                       pip_install_options: Tuple[str] = ()):
        """Prepare an archive from a user workspace."""
        # Save serialized python objects to disc
        self._serialize_interface_objects(model_provider, task_keeper, data_loader, task_assigner)
        # Save the prepared plan
        Plan.dump(Path(f'./plan/{self.plan.name}'), self.plan.config, freeze=False)

        # PACK the WORKSPACE!
        # Prepare requirements file to restore python env
        dump_requirements_file(keep_original_prefixes=True,
                               prefixes=pip_install_options)

        # Compress te workspace to restore it on collaborator
        self.arch_path = self._pack_the_workspace()

    def start(self, *, model_provider, task_keeper, data_loader,
              rounds_to_train: int,
              task_assigner=None,
              override_config: dict = None,
              delta_updates: bool = False,
              opt_treatment: str = 'RESET',
              device_assignment_policy: str = 'CPU_ONLY',
              pip_install_options: Tuple[str] = ()) -> None:
        """
        Prepare workspace distribution and send to Director.

        A successful call of this function will result in sending the experiment workspace
        to the Director service and experiment start.

        Parameters:
        model_provider - Model Interface instance.
        task_keeper - Task Interface instance.
        data_loader - Data Interface instance.
        rounds_to_train - required number of training rounds for the experiment.
        delta_updates - [bool] Tells if collaborators should send delta updates
            for the locally tuned models. If set to False, whole checkpoints will be sent.
        opt_treatment - Optimizer state treatment policy.
            Valid options: 'RESET' - reinitialize optimizer for every round,
            'CONTINUE_LOCAL' - keep local optimizer state,
            'CONTINUE_GLOBAL' - aggregate optimizer state.
        device_assignment_policy - device assignment policy.
            Valid options: 'CPU_ONLY' - device parameter passed to tasks
            will always be 'cpu',
            'CUDA_PREFERRED' - enable passing CUDA device identifiers to tasks
            by collaborators, works with cuda-device-monitor plugin equipped Envoys.
        pip_install_options - tuple of options for the remote `pip install` calls,
            example: ('-f some.website', '--no-index')
        """
        if not task_assigner:
            task_assigner = self.define_task_assigner(task_keeper, rounds_to_train)

        self._prepare_plan(model_provider, data_loader,
                           rounds_to_train,
                           delta_updates=delta_updates, opt_treatment=opt_treatment,
                           device_assignment_policy=device_assignment_policy,
                           override_config=override_config,
                           model_interface_file='model_obj.pkl',
                           tasks_interface_file='tasks_obj.pkl',
                           dataloader_interface_file='loader_obj.pkl')

        self.prepare_workspace_distribution(
            model_provider, task_keeper, data_loader,
            task_assigner,
            pip_install_options
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
            self.remove_workspace_archive()

        if response.accepted:
            self.logger.info('Experiment was accepted and launched.')
            self.experiment_accepted = True
        else:
            self.logger.info('Experiment was not accepted or failed.')

    def define_task_assigner(self, task_keeper, rounds_to_train):
        """Define task assigner by registered tasks."""
        tasks = task_keeper.get_registered_tasks()
        is_train_task_exist = False
        self.is_validate_task_exist = False
        for task in tasks.values():
            if task.task_type == 'train':
                is_train_task_exist = True
            if task.task_type == 'validate':
                self.is_validate_task_exist = True

        if not is_train_task_exist and rounds_to_train != 1:
            # Since we have only validation tasks, we do not have to train it multiple times
            raise Exception('Variable rounds_to_train must be equal 1, '
                            'because only validation tasks were given')
        if is_train_task_exist and self.is_validate_task_exist:
            def assigner(collaborators, round_number, **kwargs):
                tasks_by_collaborator = {}
                for collaborator in collaborators:
                    tasks_by_collaborator[collaborator] = [
                        tasks['train'],
                        tasks['locally_tuned_model_validate'],
                        tasks['aggregated_model_validate'],
                    ]
                return tasks_by_collaborator
            return assigner
        elif not is_train_task_exist and self.is_validate_task_exist:
            def assigner(collaborators, round_number, **kwargs):
                tasks_by_collaborator = {}
                for collaborator in collaborators:
                    tasks_by_collaborator[collaborator] = [
                        tasks['aggregated_model_validate'],
                    ]
                return tasks_by_collaborator
            return assigner
        elif is_train_task_exist and not self.is_validate_task_exist:
            raise Exception('You should define validate task!')
        else:
            raise Exception('You should define train and validate tasks!')

    def restore_experiment_state(self, model_provider):
        """Restore accepted experiment object."""
        self.task_runner_stub = self.plan.get_core_task_runner(model_provider=model_provider)
        self.current_model_status = ModelStatus.RESTORED
        self.experiment_accepted = True

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
        makedirs(tmp_dir, exist_ok=True)

        ignore = ignore_patterns(
            '__pycache__', 'data', 'cert', tmp_dir, '*.crt', '*.key',
            '*.csr', '*.srl', '*.pem', '*.pbuf', '*zip')

        copytree('./', tmp_dir + '/workspace', ignore=ignore)

        arch_path = make_archive(archive_name, archive_type, tmp_dir + '/workspace')

        rmtree(tmp_dir)

        return arch_path

    def remove_workspace_archive(self):
        """Remove the workspace archive."""
        os.remove(self.arch_path)
        del self.arch_path

    def _get_initial_tensor_dict(self, model_provider):
        """Extract initial weights from the model."""
        self.task_runner_stub = self.plan.get_core_task_runner(model_provider=model_provider)
        self.current_model_status = ModelStatus.INITIAL
        tensor_dict, _ = split_tensor_dict_for_holdouts(
            self.logger,
            self.task_runner_stub.get_tensor_dict(False),
            **self.task_runner_stub.tensor_dict_split_fn_kwargs
        )
        return tensor_dict

    def _prepare_plan(self, model_provider, data_loader,
                      rounds_to_train,
                      delta_updates, opt_treatment,
                      device_assignment_policy,
                      override_config=None,
                      model_interface_file='model_obj.pkl', tasks_interface_file='tasks_obj.pkl',
                      dataloader_interface_file='loader_obj.pkl',
                      aggregation_function_interface_file='aggregation_function_obj.pkl',
                      task_assigner_file='task_assigner_obj.pkl'):
        """Fill plan.yaml file using user provided setting."""

        # Seems like we still need to fill authorized_cols list
        # So aggregator know when to start sending tasks
        # We also could change the aggregator logic so it will send tasks to aggregator
        # as soon as it connects. This change should be a part of a bigger PR
        # brining in fault tolerance changes

        shard_registry = self.federation.get_shard_registry()
        self.plan.authorized_cols = [
            name for name, info in shard_registry.items() if info['is_online']
        ]
        # Network part of the plan
        # We keep in mind that an aggregator FQND will be the same as the directors FQDN
        # We just choose a port randomly from plan hash
        director_fqdn = self.federation.director_node_fqdn.split(':')[0]  # We drop the port
        self.plan.config['network']['settings']['agg_addr'] = director_fqdn
        self.plan.config['network']['settings']['tls'] = self.federation.tls

        # Aggregator part of the plan
        self.plan.config['aggregator']['settings']['rounds_to_train'] = rounds_to_train

        # Collaborator part
        self.plan.config['collaborator']['settings']['delta_updates'] = delta_updates
        self.plan.config['collaborator']['settings']['opt_treatment'] = opt_treatment
        self.plan.config['collaborator']['settings'][
            'device_assignment_policy'] = device_assignment_policy

        # DataLoader part
        for setting, value in data_loader.kwargs.items():
            self.plan.config['data_loader']['settings'][setting] = value

        # TaskRunner framework plugin
        # ['required_plugin_components'] should be already in the default plan with all the fields
        # filled with the default values
        self.plan.config['task_runner']['required_plugin_components'] = {
            'framework_adapters': model_provider.framework_plugin
        }

        # API layer
        self.plan.config['api_layer'] = {
            'required_plugin_components': {
                'serializer_plugin': self.serializer_plugin
            },
            'settings': {
                'model_interface_file': model_interface_file,
                'tasks_interface_file': tasks_interface_file,
                'dataloader_interface_file': dataloader_interface_file,
                'aggregation_function_interface_file': aggregation_function_interface_file,
                'task_assigner_file': task_assigner_file
            }
        }

        if override_config:
            self.plan = update_plan(override_config, plan=self.plan, resolve=False)

    def _serialize_interface_objects(
            self,
            model_provider,
            task_keeper,
            data_loader,
            task_assigner
    ):
        """Save python objects to be restored on collaborators."""
        serializer = self.plan.build(
            self.plan.config['api_layer']['required_plugin_components']['serializer_plugin'], {})
        framework_adapter = Plan.build(model_provider.framework_plugin, {})
        # Model provider serialization may need preprocessing steps
        framework_adapter.serialization_setup()

        obj_dict = {
            'model_interface_file': model_provider,
            'tasks_interface_file': task_keeper,
            'dataloader_interface_file': data_loader,
            'aggregation_function_interface_file': task_keeper.aggregation_functions,
            'task_assigner_file': task_assigner
        }

        for filename, object_ in obj_dict.items():
            serializer.serialize(object_, self.plan.config['api_layer']['settings'][filename])


class TaskKeeper:
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
        # Mapping 'task name' -> callable
        self.aggregation_functions = defaultdict(WeightedAverage)
        # Mapping 'task_alias' -> Task
        self._tasks: Dict[str, Task] = {}

    def register_fl_task(self, model, data_loader, device, optimizer=None, round_num=None):
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
            return {'metric_name': metric, 'metric_name_2': metric_2,}
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
            function_name = training_method.__name__
            self.task_registry[function_name] = wrapper_decorator
            contract = {'model': model, 'data_loader': data_loader,
                        'device': device, 'optimizer': optimizer, 'round_num': round_num}
            self.task_contract[function_name] = contract
            # define tasks
            if optimizer:
                self._tasks['train'] = TrainTask(
                    name='train',
                    function_name=function_name,
                )
            else:
                self._tasks['locally_tuned_model_validate'] = ValidateTask(
                    name='locally_tuned_model_validate',
                    function_name=function_name,
                    apply_local=True,
                )
                self._tasks['aggregated_model_validate'] = ValidateTask(
                    name='aggregated_model_validate',
                    function_name=function_name,
                )
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

    def set_aggregation_function(self, aggregation_function: AggregationFunction):
        """Set aggregation function for the task.

        To be serialized and sent to aggregator node.

        There is no support for aggregation functions
        containing logic from workspace-related libraries
        that are not present on director yet.

        Args:
            aggregation_function: Aggregation function.

        You might need to override default FedAvg aggregation with built-in aggregation types:
            - openfl.interface.aggregation_functions.GeometricMedian
            - openfl.interface.aggregation_functions.Median
        or define your own AggregationFunction subclass.
        See more details on `Overriding the aggregation function`_ documentation page.
        .. _Overriding the aggregation function:
            https://openfl.readthedocs.io/en/latest/overriding_agg_fn.html
        """
        def decorator_with_args(training_method):
            if not isinstance(aggregation_function, AggregationFunction):
                raise Exception('aggregation_function must implement '
                                'AggregationFunction interface.')
            self.aggregation_functions[training_method.__name__] = aggregation_function
            return training_method
        return decorator_with_args

    def get_registered_tasks(self) -> Dict[str, Task]:
        """Return registered tasks."""
        return self._tasks


# Backward compatibility
TaskInterface = TaskKeeper


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
