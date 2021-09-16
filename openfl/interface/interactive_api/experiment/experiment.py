# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Python low-level API module."""
import json
import os
import time

from copy import deepcopy
from logging import getLogger
from pathlib import Path

from tensorboardX import SummaryWriter

from openfl.federated import Plan
from openfl.interface.cli import setup_logging
from openfl.interface.cli_helper import WORKSPACE
from openfl.utilities import split_tensor_dict_for_holdouts


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

        self.logger = getLogger(__name__)
        setup_logging()

    def _assert_experiment_accepted(self):
        """Assure experiment is sent to director."""
        if not self.experiment_accepted:
            self.logger.error('The experimnet has not been accepted by director')
            self.logger.error(
                'Report the experiment first: '
                'use the Experiment.start() method.')
            raise Exception

    def get_best_model(self):
        """Retrieve the model with the best score."""
        self._assert_experiment_accepted()
        tensor_dict = self.federation.dir_client.get_best_model(
            experiment_name=self.experiment_name)

        return self._rebuild_model(tensor_dict)

    def get_last_model(self):
        """Retrieve the aggregated model after the last round."""
        self._assert_experiment_accepted()
        tensor_dict = self.federation.dir_client.get_last_model(
            experiment_name=self.experiment_name)

        return self._rebuild_model(tensor_dict)

    def _rebuild_model(self, tensor_dict):
        """Use tensor dict to update model weights."""
        if len(tensor_dict) == 0:
            self.logger.error('No tensors received from director')
            self.logger.error(
                'Possible reasons:\n'
                '1. Aggregated model is not ready \n'
                '2. Experiment data removed from director'
            )
        else:
            self.task_runner_stub.rebuild_model(tensor_dict, validation=True, device='cpu')

        return self.task_runner_stub.model

    def stream_metrics(self, tensorboard_logs: bool = True) -> None:
        """Stream metrics."""
        self._assert_experiment_accepted()
        for metric_message_dict in self.federation.dir_client.stream_metrics(self.experiment_name):
            self.logger.metric(
                f'Round {metric_message_dict["round"]}, '
                f'collaborator {metric_message_dict["metric_origin"]} '
                f'{metric_message_dict["task_name"]} result '
                f'{metric_message_dict["metric_name"]}:\t{metric_message_dict["metric_value"]}')

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
            self.remove_workspace_archive()

        if response.accepted:
            self.logger.info('Experiment was accepted and launched.')
            self.experiment_accepted = True
        else:
            self.logger.info('Experiment was not accepted or failed.')

    def restore_experiment_state(self, model_provider):
        """Restore accepted experimnet object."""
        self.task_runner_stub = self.plan.get_core_task_runner(model_provider=model_provider)
        self.experiment_accepted = True

    @staticmethod
    def _export_python_env():
        """Prepare requirements.txt."""
        from pip._internal.operations import freeze
        requirements_generator = freeze.freeze()

        def is_package_has_version(package: str) -> bool:
            return '==' in package and package != 'pkg-resources==0.0.0'

        with open('./requirements.txt', 'w') as f:
            for pack in requirements_generator:
                if is_package_has_version(pack):
                    f.write(pack + '\n')

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
        shard_registry = self.federation.get_shard_registry()
        plan.authorized_cols = [
            name for name, info in shard_registry.items() if info['is_online']
        ]
        # Network part of the plan
        # We keep in mind that an aggregator FQND will be the same as the directors FQDN
        # We just choose a port randomly from plan hash
        director_fqdn = self.federation.director_node_fqdn.split(':')[0]  # We drop the port
        plan.config['network']['settings']['agg_addr'] = director_fqdn
        plan.config['network']['settings']['tls'] = self.federation.tls

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
            agg_type_cls = task_keeper.aggregation_types[name].__class__
            if task_keeper.task_contract[name]['optimizer'] is not None:
                # This is training task
                plan.config['tasks'][name] = {
                    'function': name,
                    'aggregation_type': {
                        'template': f'{agg_type_cls.__module__}.{agg_type_cls.__name__}'
                    },
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
                        'aggregation_type': {
                            'template': f'{agg_type_cls.__module__}.{agg_type_cls.__name__}'
                        },
                        'kwargs': task_kwargs
                    }

        # TaskRunner framework plugin
        # ['required_plugin_components'] should be already in the default plan with all the fields
        # filled with the default values
        plan.config['task_runner']['required_plugin_components'] = {
            'framework_adapters': model_provider.framework_plugin
        }

        # API layer
        plan.config['api_layer'] = {
            'required_plugin_components': {
                'serializer_plugin': self.serializer_plugin
            },
            'settings': {
                'model_interface_file': model_interface_file,
                'tasks_interface_file': tasks_interface_file,
                'dataloader_interface_file': dataloader_interface_file,
            }
        }

        plan.config['assigner']['settings']['task_groups'][0]['tasks'] = [
            entry
            for entry in plan.config['tasks']
            if (type(plan.config['tasks'][entry]) is dict
                and 'function' in plan.config['tasks'][entry])
        ]
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
