# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Plan module."""

from hashlib import sha384
from logging import getLogger
from os.path import splitext
from importlib import import_module
from pathlib import Path
from yaml import safe_load, dump
from socket import getfqdn

from openfl.transport import AggregatorGRPCServer
from openfl.transport import CollaboratorGRPCClient

SETTINGS = 'settings'
TEMPLATE = 'template'
DEFAULTS = 'defaults'
AUTO = 'auto'


class Plan(object):
    """Federated Learning plan."""

    logger = getLogger(__name__)

    @staticmethod
    def Load(yaml_path: Path, default={}):
        """Load the plan from YAML file."""
        if yaml_path and yaml_path.exists():
            return safe_load(yaml_path.read_text())

        return default

    @staticmethod
    def Dump(yaml_path, config, freeze=False):
        """Dump the plan config to YAML file."""
        if freeze:
            plan = Plan()
            plan.config = config
            frozen_yaml_path = Path(
                f"{yaml_path.parent}/{yaml_path.stem}_{plan.hash[:8]}.yaml")
            if frozen_yaml_path.exists():
                Plan.logger.info(f"{yaml_path.name} is already frozen")
                return
            frozen_yaml_path.write_text(dump(config))
            frozen_yaml_path.chmod(0o400)
            Plan.logger.info(f"{yaml_path.name} frozen successfully")
        else:
            yaml_path.write_text(dump(config))

    @staticmethod
    def Parse(plan_config_path: Path, cols_config_path: Path = None,
              data_config_path: Path = None, resolve=True):
        """
        Parse the Federated Learning plan.

        Args:
            plan_config_path (string): The filepath to the federated learning
                                       plan
            cols_config_path (string): The filepath to the federation
                                       collaborator list [optional]
            data_config_path (string): The filepath to the federation
                                       collaborator data configuration
                                       [optional]
        Returns:
            A federated learning plan object
        """
        try:

            plan = Plan()
            plan.config = Plan.Load(plan_config_path)  # load plan configuration
            plan.name = plan_config_path.name
            plan.files = [plan_config_path]  # collect all the plan files

            # ensure 'settings' appears in each top-level section
            for section in plan.config.keys():

                if plan.config[section].get(SETTINGS) is None:
                    plan.config[section][SETTINGS] = {}

            # walk the top level keys and load 'defaults' in sorted order
            for section in sorted(plan.config.keys()):
                defaults = plan.config[section].get(DEFAULTS)

                if defaults is not None:

                    plan.files.append(defaults)

                    if resolve:
                        Plan.logger.info(
                            f'Loading DEFAULTS for section [red]{section}[/] '
                            f'from file [red]{defaults}[/].',
                            extra={'markup': True})

                    defaults = Plan.Load(Path(defaults))

                    if SETTINGS in defaults:
                        # override defaults with section settings
                        defaults[SETTINGS].update(
                            plan.config[section][SETTINGS])
                        plan.config[section][SETTINGS] = defaults[SETTINGS]

                    defaults.update(plan.config[section])

                    plan.config[section] = defaults

            plan.authorized_cols = Plan.Load(cols_config_path).get(
                'collaborators', []
            )

            # TODO: Does this need to be a YAML file? Probably want to use key
            #  value as the plan hash
            plan.cols_data_paths = {}
            if data_config_path is not None:
                data_config = open(data_config_path, "r")
                for line in data_config:
                    line = line.rstrip()
                    if len(line) > 0:
                        if line[0] != '#':
                            collab, data_path = line.split(',')
                            plan.cols_data_paths[collab] = data_path

            if resolve:
                plan.resolve()

                Plan.logger.info(
                    f'Parsing Federated Learning Plan : [green]SUCCESS[/] : '
                    f'[blue]{plan_config_path}[/].',
                    extra={'markup': True})
                Plan.logger.info(dump(plan.config))

            return plan

        except Exception:
            Plan.logger.error(f'Parsing Federated Learning Plan : '
                              f'[red]FAILURE[/] : [blue]{plan_config_path}[/].',
                              extra={'markup': True})
            raise

    @staticmethod
    def Build(template, settings, **override):
        """
        Create an instance of a openfl Component or Federated DataLoader/TaskRunner.

        Args:
            template: Fully qualified class template path
            settings: Keyword arguments to class constructor

        Returns:
            A Python object
        """
        # from sys import path

        # for x in path:
        #     Plan.logger.info(f'sys.path: {x}')

        class_name = splitext(template)[1].strip('.')
        module_path = splitext(template)[0]

        Plan.logger.info(f'Building [red]🡆[/] Object [red]{class_name}[/] '
                         f'from [red]{module_path}[/] Module.',
                         extra={'markup': True})
        Plan.logger.info(f'Settings [red]🡆[/] {settings}',
                         extra={'markup': True})
        Plan.logger.info(f'Override [red]🡆[/] {override}',
                         extra={'markup': True})

        settings.update(**override)

        module = import_module(module_path)
        instance = getattr(module, class_name)(**settings)

        return instance

    def __init__(self):
        """Initialize."""
        self.config = {}  # dictionary containing patched plan definition
        self.authorized_cols = []  # authorized collaborator list
        self.cols_data_paths = {}  # collaborator data paths dict

        self.collaborator_ = None  # collaborator object
        self.aggregator_ = None  # aggregator object
        self.assigner_ = None  # assigner object

        self.loader_ = None  # data loader object
        self.runner_ = None  # task runner object

        self.server_ = None  # gRPC server object
        self.client_ = None  # gRPC client object

        self.pipe_ = None  # compression pipeline object

        self.hash_ = None
        self.name_ = None

    @property
    def hash(self):
        """Generate hash for this instance."""
        self.hash_ = sha384(dump(self.config).encode('utf-8'))
        Plan.logger.info(f'FL-Plan hash is [blue]{self.hash_.hexdigest()}[/]',
                         extra={'markup': True})

        return self.hash_.hexdigest()

    def resolve(self):
        """Resolve the federation settings."""
        self.federation_uuid = f'{self.name}_{self.hash[:8]}'
        self.aggregator_uuid = f'aggregator_{self.federation_uuid}'

        self.rounds_to_train = self.config['aggregator'][SETTINGS][
            'rounds_to_train']

        if self.config['network'][SETTINGS]['agg_addr'] == AUTO:
            self.config['network'][SETTINGS]['agg_addr'] = getfqdn()

        if self.config['network'][SETTINGS]['agg_port'] == AUTO:
            self.config['network'][SETTINGS]['agg_port'] = int(
                self.hash[:8], 16
            ) % (60999 - 49152) + 49152

    def get_assigner(self):
        """Get the plan task assigner."""
        defaults = self.config.get('assigner',
                                   {
                                       TEMPLATE: 'openfl.component.Assigner',
                                       SETTINGS: {}
                                   })

        defaults[SETTINGS]['authorized_cols'] = self.authorized_cols
        defaults[SETTINGS]['rounds_to_train'] = self.rounds_to_train
        defaults[SETTINGS]['tasks'] = self.config.get('tasks', {})

        if self.assigner_ is None:
            self.assigner_ = Plan.Build(**defaults)

        return self.assigner_

    def get_aggregator(self):
        """Get federation aggregator."""
        defaults = self.config.get('aggregator',
                                   {
                                       TEMPLATE: 'openfl.component.Aggregator',
                                       SETTINGS: {}
                                   })

        defaults[SETTINGS]['aggregator_uuid'] = self.aggregator_uuid
        defaults[SETTINGS]['federation_uuid'] = self.federation_uuid
        defaults[SETTINGS]['authorized_cols'] = self.authorized_cols
        defaults[SETTINGS]['assigner'] = self.get_assigner()

        if self.aggregator_ is None:
            self.aggregator_ = Plan.Build(**defaults)

        return self.aggregator_

    def get_tensor_pipe(self):
        """Get data tensor pipeline."""
        defaults = self.config.get(
            'compression_pipeline',
            {
                TEMPLATE: 'openfl.pipelines.NoCompressionPipeline',
                SETTINGS: {}
            }
        )

        if self.pipe_ is None:
            self.pipe_ = Plan.Build(**defaults)

        return self.pipe_

    def get_data_loader(self, collaborator_name):
        """Get data loader."""
        defaults = self.config.get('data_loader',
                                   {
                                       TEMPLATE: 'openfl.federation.DataLoader',
                                       SETTINGS: {}
                                   })

        defaults[SETTINGS]['data_path'] = self.cols_data_paths[
            collaborator_name
        ]

        if self.loader_ is None:
            self.loader_ = Plan.Build(**defaults)

        return self.loader_

    def get_task_runner(self, collaborator_name):
        """Get task runner."""
        defaults = self.config.get('task_runner',
                                   {
                                       TEMPLATE: 'openfl.federation.TaskRunner',
                                       SETTINGS: {}
                                   })

        defaults[SETTINGS]['data_loader'] = self.get_data_loader(
            collaborator_name
        )

        if self.runner_ is None:
            self.runner_ = Plan.Build(**defaults)

        return self.runner_

    def get_collaborator(self, collaborator_name,
                         task_runner=None, client=None):
        """Get collaborator."""
        defaults = self.config.get(
            'collaborator',
            {
                TEMPLATE: 'openfl.component.Collaborator',
                SETTINGS: {}
            }
        )

        defaults[SETTINGS]['collaborator_name'] = collaborator_name
        defaults[SETTINGS]['aggregator_uuid'] = self.aggregator_uuid
        defaults[SETTINGS]['federation_uuid'] = self.federation_uuid
        if task_runner is not None:
            defaults[SETTINGS]['task_runner'] = task_runner
        else:
            defaults[SETTINGS]['task_runner'] = self.get_task_runner(
                collaborator_name
            )
        defaults[SETTINGS]['tensor_pipe'] = self.get_tensor_pipe()
        defaults[SETTINGS]['task_config'] = self.config.get('tasks', {})
        if client is not None:
            defaults[SETTINGS]['client'] = client
        else:
            defaults[SETTINGS]['client'] = self.get_client(
                collaborator_name,
                self.aggregator_uuid,
                self.federation_uuid
            )

        if self.collaborator_ is None:
            self.collaborator_ = Plan.Build(**defaults)

        return self.collaborator_

    def get_client(self, collaborator_name, aggregator_uuid, federation_uuid):
        """Get gRPC client for the specified collaborator."""
        common_name = collaborator_name

        chain = 'cert/cert_chain.crt'
        certificate = f'cert/client/col_{common_name}.crt'
        private_key = f'cert/client/col_{common_name}.key'

        client_args = self.config['network'][SETTINGS]

        # patch certificates

        client_args['ca'] = chain
        client_args['certificate'] = certificate
        client_args['private_key'] = private_key

        client_args['aggregator_uuid'] = aggregator_uuid
        client_args['federation_uuid'] = federation_uuid

        if self.client_ is None:
            self.client_ = CollaboratorGRPCClient(**client_args)

        return self.client_

    def get_server(self):
        """Get gRPC server of the aggregator instance."""
        common_name = self.config['network'][SETTINGS]['agg_addr'].lower()

        chain = 'cert/cert_chain.crt'
        certificate = f'cert/server/agg_{common_name}.crt'
        private_key = f'cert/server/agg_{common_name}.key'

        server_args = self.config['network'][SETTINGS]

        # patch certificates

        server_args['ca'] = chain
        server_args['certificate'] = certificate
        server_args['private_key'] = private_key

        server_args['aggregator'] = self.get_aggregator()

        if self.server_ is None:
            self.server_ = AggregatorGRPCServer(**server_args)

        return self.server_
