# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Plan module."""
from openfl.utilities.utils import getfqdn_env

# from openfl.transport import AggregatorGRPCServer
# from openfl.transport import AggregatorGRPCClient
from openfl.experimental.transport import AggregatorGRPCServer
from openfl.experimental.transport import AggregatorGRPCClient

from openfl.interface.cli_helper import WORKSPACE
from openfl.component.assigner.custom_assigner import Assigner
from openfl.interface.aggregation_functions import WeightedAverage
from openfl.interface.aggregation_functions import AggregationFunction
from yaml import SafeDumper
from yaml import safe_load
from yaml import dump
from pathlib import Path
from os.path import splitext
from logging import getLogger
from importlib import import_module
from hashlib import sha384
import importlib
import sys
sys.path.append(
    "/home/parth-wsl/env-ishant-code/openfl/openfl-workspace/experimental-wi/testflow.py")


SETTINGS = 'settings'
TEMPLATE = 'template'
DEFAULTS = 'defaults'
AUTO = 'auto'


class Plan:
    """Federated Learning plan."""

    logger = getLogger(__name__)

    @staticmethod
    def load(yaml_path: Path, default: dict = None):
        """Load the plan from YAML file."""
        if default is None:
            default = {}
        if yaml_path and yaml_path.exists():
            return safe_load(yaml_path.read_text())
        return default

    @staticmethod
    def dump(yaml_path, config, freeze=False):
        """Dump the plan config to YAML file."""

        class NoAliasDumper(SafeDumper):

            def ignore_aliases(self, data):
                return True

        if freeze:
            plan = Plan()
            plan.config = config
            frozen_yaml_path = Path(
                f'{yaml_path.parent}/{yaml_path.stem}_{plan.hash[:8]}.yaml')
            if frozen_yaml_path.exists():
                Plan.logger.info(f'{yaml_path.name} is already frozen')
                return
            frozen_yaml_path.write_text(dump(config))
            frozen_yaml_path.chmod(0o400)
            Plan.logger.info(f'{yaml_path.name} frozen successfully')
        else:
            yaml_path.write_text(dump(config))

    @staticmethod
    def parse(plan_config_path: Path, cols_config_path: Path = None,
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
            # load plan configuration
            plan.config = Plan.load(plan_config_path)
            plan.name = plan_config_path.name
            plan.files = [plan_config_path]  # collect all the plan files

            # ensure 'settings' appears in each top-level section
            for section in plan.config.keys():

                if plan.config[section].get(SETTINGS) is None:
                    plan.config[section][SETTINGS] = {}

            # walk the top level keys and load 'defaults' in sorted order
            for section in sorted(plan.config.keys()):
                defaults = plan.config[section].pop(DEFAULTS, None)

                if defaults is not None:
                    defaults = WORKSPACE / 'workspace' / defaults

                    plan.files.append(defaults)

                    if resolve:
                        Plan.logger.info(
                            f'Loading DEFAULTS for section [red]{section}[/] '
                            f'from file [red]{defaults}[/].',
                            extra={'markup': True})

                    defaults = Plan.load(Path(defaults))

                    if SETTINGS in defaults:
                        # override defaults with section settings
                        defaults[SETTINGS].update(
                            plan.config[section][SETTINGS])
                        plan.config[section][SETTINGS] = defaults[SETTINGS]

                    defaults.update(plan.config[section])

                    plan.config[section] = defaults

            plan.authorized_cols = Plan.load(cols_config_path).get(
                'collaborators', []
            )

            # TODO: Does this need to be a YAML file? Probably want to use key
            #  value as the plan hash
            plan.cols_data_paths = {}
            if data_config_path is not None:
                data_config = open(data_config_path, 'r')
                for line in data_config:
                    line = line.rstrip()
                    if len(line) > 0:
                        if line[0] != '#':
                            collab, data_path = line.split(',', maxsplit=1)
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
            Plan.logger.exception(f'Parsing Federated Learning Plan : '
                                  f'[red]FAILURE[/] : [blue]{plan_config_path}[/].',
                                  extra={'markup': True})
            raise

    @staticmethod
    def build(template, settings, **override):
        """
        Create an instance of a openfl Component or Federated DataLoader/TaskRunner.

        Args:
            template: Fully qualified class template path
            settings: Keyword arguments to class constructor

        Returns:
            A Python object
        """
        class_name = splitext(template)[1].strip('.')
        module_path = splitext(template)[0]

        Plan.logger.info(f'Building [red]🡆[/] Object [red]{class_name}[/] '
                         f'from [red]{module_path}[/] Module.',
                         extra={'markup': True})
        Plan.logger.debug(f'Settings [red]🡆[/] {settings}',
                          extra={'markup': True})
        Plan.logger.debug(f'Override [red]🡆[/] {override}',
                          extra={'markup': True})

        settings.update(**override)

        module = import_module(module_path)
        instance = getattr(module, class_name)(**settings)

        return instance

    @staticmethod
    def import_(template):
        """
        Import an instance of a openfl Component or Federated DataLoader/TaskRunner.

        Args:
            template: Fully qualified object path

        Returns:
            A Python object
        """
        class_name = splitext(template)[1].strip('.')
        module_path = splitext(template)[0]
        Plan.logger.info(f'Importing [red]🡆[/] Object [red]{class_name}[/] '
                         f'from [red]{module_path}[/] Module.',
                         extra={'markup': True})
        module = import_module(module_path)
        instance = getattr(module, class_name)

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

        self.straggler_policy_ = None  # straggler handling policy

        self.hash_ = None
        self.name_ = None
        self.serializer_ = None

    @property
    def hash(self):  # NOQA
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
            self.config['network'][SETTINGS]['agg_addr'] = getfqdn_env()

        if self.config['network'][SETTINGS]['agg_port'] == AUTO:
            self.config['network'][SETTINGS]['agg_port'] = int(
                self.hash[:8], 16
            ) % (60999 - 49152) + 49152

    def get_assigner(self):
        """Get the plan task assigner."""
        aggregation_functions_by_task = None
        assigner_function = None
        try:
            aggregation_functions_by_task = self.restore_object(
                'aggregation_function_obj.pkl')
            assigner_function = self.restore_object('task_assigner_obj.pkl')
        except Exception as exc:
            self.logger.error(
                f'Failed to load aggregation and assigner functions: {exc}')
            self.logger.info('Using Task Runner API workflow')
        if assigner_function:
            self.assigner_ = Assigner(
                assigner_function=assigner_function,
                aggregation_functions_by_task=aggregation_functions_by_task,
                authorized_cols=self.authorized_cols,
                rounds_to_train=self.rounds_to_train,
            )
        else:
            # Backward compatibility
            defaults = self.config.get(
                'assigner',
                {
                    TEMPLATE: 'openfl.component.Assigner',
                    SETTINGS: {}
                }
            )

            defaults[SETTINGS]['authorized_cols'] = self.authorized_cols
            defaults[SETTINGS]['rounds_to_train'] = self.rounds_to_train
            defaults[SETTINGS]['tasks'] = self.get_tasks()

            if self.assigner_ is None:
                self.assigner_ = Plan.build(**defaults)

        return self.assigner_

    def get_tasks(self):
        """Get federation tasks."""
        tasks = self.config.get('tasks', {})
        tasks.pop(DEFAULTS, None)
        tasks.pop(SETTINGS, None)
        for task in tasks:
            aggregation_type = tasks[task].get('aggregation_type')
            if aggregation_type is None:
                aggregation_type = WeightedAverage()
            elif isinstance(aggregation_type, dict):
                if SETTINGS not in aggregation_type:
                    aggregation_type[SETTINGS] = {}
                aggregation_type = Plan.build(**aggregation_type)
                if not isinstance(aggregation_type, AggregationFunction):
                    raise NotImplementedError(
                        f'''{task} task aggregation type does not implement an interface:
    openfl.interface.aggregation_functions.AggregationFunction
    ''')
            tasks[task]['aggregation_type'] = aggregation_type
        return tasks

    def get_aggregator(self, tensor_dict=None):
        """Get federation aggregator."""
        import sys
        sys.path.append("/home/parth-wsl/env-ishant-code/openfl/" +
                        "openfl-workspace/mnist/src/aggregator_private_attrs.py")
        sys.path.append("/home/parth-wsl/env-ishant-code/openfl/" +
                        "openfl-workspace/mnist/src/mnist.py")

        defaults = self.config.get('experimental.aggregator', # modifying to not to get values from yaml file
                                   {
                                       TEMPLATE: 'openfl.experimental.component.Aggregator',
                                       SETTINGS: {}
                                   })
        defaults[SETTINGS]['name'] = "agg"
        defaults[SETTINGS]['aggregator_uuid'] = self.aggregator_uuid
        defaults[SETTINGS]['federation_uuid'] = self.federation_uuid
        defaults[SETTINGS]['authorized_cols'] = self.authorized_cols

        # testflow_module = importlib.import_module("src.mnist")
        # TODO: Get checkpoint from plan.yaml
        defaults[SETTINGS]['checkpoint'] = False

        # Get flow class, and initialize it here.
        self.logger.info("Initialzing flow.")
        defaults[SETTINGS]['flow'] = getattr(
            importlib.import_module("src.mnist"),
            "FederatedFlow"
        )(checkpoint=defaults[SETTINGS]['checkpoint'])

        # Initializing runtime - this will be moved to \
        # experimental.component.aggregator.py
        defaults[SETTINGS]['runtime'] = getattr(
            importlib.import_module("openfl.experimental.runtime"),
            "FederatedRuntime"
        )()

        # Getting aggregator private attrs function and it's arguments
        defaults[SETTINGS]['private_attrs_callable'] = getattr(
            importlib.import_module("src.aggregator_private_attrs"),
            "aggregator_private_attrs"
        )
        defaults[SETTINGS]['private_attrs_kwargs'] = getattr(
            importlib.import_module("src.aggregator_private_attrs"),
            "aggregator_kwargs"
        )

        defaults[SETTINGS]['compression_pipeline'] = self.get_tensor_pipe()
        log_metric_callback = defaults[SETTINGS].get('log_metric_callback')

        if log_metric_callback:
            if isinstance(log_metric_callback, dict):
                log_metric_callback = Plan.import_(**log_metric_callback)
            elif not callable(log_metric_callback):
                raise TypeError(f'log_metric_callback should be callable object '
                                f'or be import from code part, get {log_metric_callback}')

        defaults[SETTINGS]['log_metric_callback'] = log_metric_callback
        if self.aggregator_ is None:
            self.aggregator_ = Plan.build(
                **defaults, initial_tensor_dict=tensor_dict)

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
            self.pipe_ = Plan.build(**defaults)

        return self.pipe_

    def get_straggler_handling_policy(self):
        """Get straggler handling policy."""
        template = 'openfl.component.straggler_handling_functions.CutoffTimeBasedStragglerHandling'
        defaults = self.config.get(
            'straggler_handling_policy',
            {
                TEMPLATE: template,
                SETTINGS: {}
            }
        )

        if self.straggler_policy_ is None:
            self.straggler_policy_ = Plan.build(**defaults)

        return self.straggler_policy_

    # legacy api (TaskRunner subclassing)
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
            self.loader_ = Plan.build(**defaults)

        return self.loader_

    # Python interactive api
    def initialize_data_loader(self, data_loader, shard_descriptor):
        """Get data loader."""
        data_loader.shard_descriptor = shard_descriptor
        return data_loader

    # legacy api (TaskRunner subclassing)
    def get_task_runner(self, data_loader):
        """Get task runner."""
        defaults = self.config.get('task_runner',
                                   {
                                       TEMPLATE: 'openfl.federation.TaskRunner',
                                       SETTINGS: {}
                                   })

        defaults[SETTINGS]['data_loader'] = data_loader

        if self.runner_ is None:
            self.runner_ = Plan.build(**defaults)

        return self.runner_

    # Python interactive api
    def get_core_task_runner(self, data_loader=None,
                             model_provider=None,
                             task_keeper=None):
        """Get task runner."""
        defaults = self.config.get(
            'task_runner',
            {
                TEMPLATE: 'openfl.federated.task.task_runner.CoreTaskRunner',
                SETTINGS: {}
            })

        # We are importing a CoreTaskRunner instance!!!
        if self.runner_ is None:
            self.runner_ = Plan.build(**defaults)

        self.runner_.set_data_loader(data_loader)

        self.runner_.set_model_provider(model_provider)
        self.runner_.set_task_provider(task_keeper)

        framework_adapter = Plan.build(
            self.config['task_runner']['required_plugin_components']['framework_adapters'], {})

        # This step initializes tensorkeys
        # Which have no sens if task provider is not set up
        self.runner_.set_framework_adapter(framework_adapter)

        return self.runner_

    # mnist workspace hardcode
    def get_collaborator(self, collaborator_name, root_certificate=None, private_key=None,
                         certificate=None, task_runner=None, client=None, shard_descriptor=None):
        """Get collaborator."""
        import sys
        sys.path.append("/home/parth-wsl/env-ishant-code/openfl/" +
                        "openfl-workspace/mnist/src/collaborator_private_attrs.py")

        defaults = self.config.get(
            'experimental.collaborator',
            {
                TEMPLATE: 'openfl.experimental.component.Collaborator',
                SETTINGS: {}
            }
        )

        defaults[SETTINGS]['collaborator_name'] = collaborator_name
        defaults[SETTINGS]['aggregator_uuid'] = self.aggregator_uuid
        defaults[SETTINGS]['federation_uuid'] = self.federation_uuid

        defaults[SETTINGS]['compression_pipeline'] = self.get_tensor_pipe()

        if client is not None:
            defaults[SETTINGS]['client'] = client
        else:
            defaults[SETTINGS]['client'] = self.get_client(
                collaborator_name,
                self.aggregator_uuid,
                self.federation_uuid,
                root_certificate,
                private_key,
                certificate
            )

        # testflow_module = importlib.import_module("src.mnist")

        # Get collaborator private attribute function and it's arguments
        defaults[SETTINGS]['private_attrs_callable'] = getattr(
            importlib.import_module("src.collaborator_private_attrs"),
            "collaborator_private_attrs"
        )

        defaults[SETTINGS]['private_attrs_kwargs'] = getattr(
            importlib.import_module("src.collaborator_private_attrs"),
            "collaborator_kwargs"
        )

        if self.collaborator_ is None:
            self.collaborator_ = Plan.build(**defaults)

        return self.collaborator_

    # testflow workflow hardcode
    # def get_collaborator(self, collaborator_name, root_certificate=None, private_key=None,
    #                      certificate=None, task_runner=None, client=None, shard_descriptor=None):
    #     """Get collaborator."""
    #     import sys
    #     sys.path.append("/home/parth-wsl/env-ishant-code/openfl/" +
    #                     "openfl-workspace/test/src/")

    #     defaults = self.config.get(
    #         'experimental.collaborator',
    #         {
    #             TEMPLATE: 'openfl.experimental.component.Collaborator',
    #             SETTINGS: {}
    #         }
    #     )

    #     defaults[SETTINGS]['collaborator_name'] = collaborator_name
    #     defaults[SETTINGS]['aggregator_uuid'] = self.aggregator_uuid
    #     defaults[SETTINGS]['federation_uuid'] = self.federation_uuid

    #     defaults[SETTINGS]['compression_pipeline'] = self.get_tensor_pipe()

    #     if client is not None:
    #         defaults[SETTINGS]['client'] = client
    #     else:
    #         defaults[SETTINGS]['client'] = self.get_client(
    #             collaborator_name,
    #             self.aggregator_uuid,
    #             self.federation_uuid,
    #             root_certificate,
    #             private_key,
    #             certificate
    #         )

    #     # def collaborator_private_attrs(collab_name):
    #     #     return {
    #     #         "name": f"private attributes of {collab_name}"
    #     #     }

    #     testflow_module = importlib.import_module("src.testflow")

    #     defaults[SETTINGS]['private_attrs_callable'] = getattr(
    #         testflow_module, "collaborator_private_attrs")
    #     # defaults[SETTINGS]['private_attrs_callable'] = collaborator_private_attrs
    #     defaults[SETTINGS]['private_attrs_kwargs'] = {
    #         "collab_name": defaults[SETTINGS]['collaborator_name']
    #     }
    #     # defaults[SETTINGS]['private_attrs_kwargs'] = {
    #     #     "collab_name": defaults[SETTINGS]['collaborator_name']
    #     # }

    #     if self.collaborator_ is None:
    #         self.collaborator_ = Plan.build(**defaults)

    #     return self.collaborator_

    def get_client(self, collaborator_name, aggregator_uuid, federation_uuid,
                   root_certificate=None, private_key=None, certificate=None):
        """Get gRPC client for the specified collaborator."""
        common_name = collaborator_name
        if not root_certificate or not private_key or not certificate:
            root_certificate = 'cert/cert_chain.crt'
            certificate = f'cert/client/col_{common_name}.crt'
            private_key = f'cert/client/col_{common_name}.key'

        client_args = self.config['network'][SETTINGS]

        # patch certificates

        client_args['root_certificate'] = root_certificate
        client_args['certificate'] = certificate
        client_args['private_key'] = private_key

        client_args['aggregator_uuid'] = aggregator_uuid
        client_args['federation_uuid'] = federation_uuid

        if self.client_ is None:
            self.client_ = AggregatorGRPCClient(**client_args)

        return self.client_

    def get_server(self, root_certificate=None, private_key=None, certificate=None, **kwargs):
        """Get gRPC server of the aggregator instance."""
        common_name = self.config['network'][SETTINGS]['agg_addr'].lower()

        if not root_certificate or not private_key or not certificate:
            root_certificate = 'cert/cert_chain.crt'
            certificate = f'cert/server/agg_{common_name}.crt'
            private_key = f'cert/server/agg_{common_name}.key'

        server_args = self.config['network'][SETTINGS]

        # patch certificates

        server_args.update(kwargs)
        server_args['root_certificate'] = root_certificate
        server_args['certificate'] = certificate
        server_args['private_key'] = private_key

        aggregator_ = self.get_aggregator()
        server_args['aggregator'] = aggregator_

        if self.server_ is None:
            self.server_ = AggregatorGRPCServer(**server_args)

        return self.server_

    def interactive_api_get_server(self, *, tensor_dict, root_certificate, certificate,
                                   private_key, tls):
        """Get gRPC server of the aggregator instance."""
        server_args = self.config['network'][SETTINGS]

        # patch certificates
        server_args['root_certificate'] = root_certificate
        server_args['certificate'] = certificate
        server_args['private_key'] = private_key
        server_args['tls'] = tls

        server_args['aggregator'] = self.get_aggregator(tensor_dict)

        if self.server_ is None:
            self.server_ = AggregatorGRPCServer(**server_args)

        return self.server_

    def deserialize_interface_objects(self):
        """Deserialize objects for TaskRunner."""
        api_layer = self.config['api_layer']
        filenames = [
            'model_interface_file',
            'tasks_interface_file',
            'dataloader_interface_file'
        ]
        return (self.restore_object(api_layer['settings'][filename]) for filename in filenames)

    def get_serializer_plugin(self, **kwargs):
        """Get serializer plugin.

        This plugin is used for serialization of interfaces in new interactive API
        """
        if self.serializer_ is None:
            if 'api_layer' not in self.config:  # legacy API
                return None
            required_plugin_components = self.config['api_layer']['required_plugin_components']
            serializer_plugin = required_plugin_components['serializer_plugin']
            self.serializer_ = Plan.build(serializer_plugin, kwargs)
        return self.serializer_

    def restore_object(self, filename):
        """Deserialize an object."""
        serializer_plugin = self.get_serializer_plugin()
        if serializer_plugin is None:
            return None
        obj = serializer_plugin.restore_object(filename)
        return obj
