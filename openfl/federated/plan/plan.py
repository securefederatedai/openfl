# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Plan module."""

from functools import partial
from hashlib import sha384
from importlib import import_module
from logging import getLogger
from os.path import splitext
from pathlib import Path

from yaml import SafeDumper, dump, safe_load

from openfl.interface.aggregation_functions import AggregationFunction, WeightedAverage
from openfl.interface.cli_helper import WORKSPACE
from openfl.transport import AggregatorGRPCClient, AggregatorGRPCServer
from openfl.utilities.utils import getfqdn_env

SETTINGS = "settings"
TEMPLATE = "template"
DEFAULTS = "defaults"
AUTO = "auto"

logger = getLogger(__name__)


class Plan:
    """A class used to represent a Federated Learning plan.

    This class provides methods to manage and manipulate federated learning
    plans.

    Attributes:
        config (dict): Dictionary containing patched plan definition.
        authorized_cols (list): Authorized collaborator list.
        cols_data_paths (dict): Collaborator data paths dictionary.
        collaborator_ (Collaborator): Collaborator object.
        aggregator_ (Aggregator): Aggregator object.
        assigner_ (Assigner): Assigner object.
        loader_ (DataLoader): Data loader object.
        runner_ (TaskRunner): Task runner object.
        server_ (AggregatorGRPCServer): gRPC server object.
        client_ (AggregatorGRPCClient): gRPC client object.
        pipe_ (CompressionPipeline): Compression pipeline object.
        straggler_policy_ (StragglerPolicy): Straggler handling policy.
        hash_ (str): Hash of the instance.
        name_ (str): Name of the instance.
        serializer_ (SerializerPlugin): Serializer plugin.
    """

    @staticmethod
    def load(yaml_path: Path, default: dict = None):
        """Load the plan from YAML file.

        Args:
            yaml_path (Path): Path to the YAML file.
            default (dict, optional): Default plan configuration.
                Defaults to {}.

        Returns:
            dict: Plan configuration loaded from the YAML file.
        """
        if default is None:
            default = {}
        if yaml_path and yaml_path.exists():
            return safe_load(yaml_path.read_text())
        return default

    @staticmethod
    def dump(yaml_path, config, freeze=False):
        """Dump the plan config to YAML file.

        Args:
            yaml_path (Path): Path to the YAML file.
            config (dict): Plan configuration to be dumped.
            freeze (bool, optional): Flag to freeze the plan. Defaults to
                False.
        """

        class NoAliasDumper(SafeDumper):
            def ignore_aliases(self, data):
                return True

        if freeze:
            plan = Plan()
            plan.config = config
            frozen_yaml_path = Path(f"{yaml_path.parent}/{yaml_path.stem}_{plan.hash[:8]}.yaml")
            if frozen_yaml_path.exists():
                logger.info("%s is already frozen", yaml_path.name)
                return
            frozen_yaml_path.write_text(dump(config))
            frozen_yaml_path.chmod(0o400)
            logger.info("%s frozen successfully", yaml_path.name)
        else:
            yaml_path.write_text(dump(config))

    @staticmethod
    def parse(
        plan_config_path: Path,
        cols_config_path: Path = None,
        data_config_path: Path = None,
        gandlf_config_path: Path = None,
        resolve: bool = True,
    ):
        """
        Parse the Federated Learning plan.

        Args:
            plan_config_path (Path): The filepath to the Federated Learning
                plan.
            cols_config_path (Path, optional): The filepath to the Federation
                collaborator list. Defaults to None.
            data_config_path (Path, optional): The filepath to the Federation
                collaborator data configuration. Defaults to None.
            gandlf_config_path (Path, optional): The filepath to a yaml file
                that overrides the configuration. Defaults to None.
            resolve (bool, optional): Flag to resolve the plan settings.
                Defaults to True.

        Returns:
            Plan: A Federated Learning plan object.
        """
        try:
            plan = Plan()
            plan.config = Plan.load(plan_config_path)  # load plan configuration
            plan.name = plan_config_path.name
            plan.files = [plan_config_path]  # collect all the plan files

            Plan._ensure_settings_in_sections(plan)
            Plan._load_defaults(plan, resolve)

            if gandlf_config_path is not None:
                Plan._import_gandlf_config(plan, gandlf_config_path)

            plan.authorized_cols = Plan.load(cols_config_path).get("collaborators", [])

            Plan._load_collaborator_data_paths(plan, data_config_path)
            plan.verify()

            if resolve:
                plan.resolve()
                logger.info(
                    f"Parsing Federated Learning Plan : [green]SUCCESS[/] : "
                    f"[blue]{plan_config_path}[/].",
                    extra={"markup": True},
                )
                logger.info(dump(plan.config))

            return plan

        except Exception:
            logger.exception(
                f"Parsing Federated Learning Plan : [red]FAILURE[/] : [blue]{plan_config_path}[/].",
                extra={"markup": True},
            )
            raise

    @staticmethod
    def _ensure_settings_in_sections(plan):
        """Ensure 'settings' appears in each top-level section."""
        for section in plan.config.keys():
            if plan.config[section].get(SETTINGS) is None:
                plan.config[section][SETTINGS] = {}

    @staticmethod
    def _load_defaults(plan, resolve):
        """Load 'defaults' in sorted order for each top-level section."""
        for section in sorted(plan.config.keys()):
            defaults = plan.config[section].pop(DEFAULTS, None)

            if defaults is not None:
                defaults = WORKSPACE / "workspace" / defaults
                plan.files.append(defaults)

                if resolve:
                    logger.info(
                        f"Loading DEFAULTS for section [red]{section}[/] "
                        f"from file [red]{defaults}[/].",
                        extra={"markup": True},
                    )

                defaults = Plan.load(Path(defaults))

                if SETTINGS in defaults:
                    # override defaults with section settings
                    defaults[SETTINGS].update(plan.config[section][SETTINGS])
                    plan.config[section][SETTINGS] = defaults[SETTINGS]

                defaults.update(plan.config[section])
                plan.config[section] = defaults

    @staticmethod
    def _import_gandlf_config(plan, gandlf_config_path):
        """Import GaNDLF Config into the plan."""
        logger.info(
            f"Importing GaNDLF Config into plan from file [red]{gandlf_config_path}[/].",
            extra={"markup": True},
        )

        gandlf_config = Plan.load(Path(gandlf_config_path))
        # check for some defaults
        gandlf_config["output_dir"] = gandlf_config.get("output_dir", ".")
        plan.config["task_runner"]["settings"]["gandlf_config"] = gandlf_config

    @staticmethod
    def _load_collaborator_data_paths(plan, data_config_path):
        """Load collaborator data paths from the data configuration file."""
        plan.cols_data_paths = {}
        if data_config_path is not None:
            with open(data_config_path, "r") as data_config:
                for line in data_config:
                    line = line.rstrip()
                    if len(line) > 0 and line[0] != "#":
                        collab, data_path = line.split(",", maxsplit=1)
                        plan.cols_data_paths[collab] = data_path

    @staticmethod
    def build(template, settings, **override):
        """Create an instance of a openfl Component or Federated
        DataLoader/TaskRunner.

        Args:
            template (str): Fully qualified class template path.
            settings (dict): Keyword arguments to class constructor.
            override (dict): Additional settings to override the default ones.

        Returns:
            object: A Python object.
        """
        class_name = splitext(template)[1].strip(".")
        module_path = splitext(template)[0]

        logger.info("Building `%s` Module.", template)
        logger.debug("Settings %s", settings)
        logger.debug("Override %s", override)

        settings.update(**override)

        module = import_module(module_path)
        instance = getattr(module, class_name)(**settings)

        return instance

    @staticmethod
    def import_(template):
        """Import an instance of a openfl Component or Federated
        DataLoader/TaskRunner.

        Args:
            template (str): Fully qualified object path.

        Returns:
            object: A Python object.
        """
        class_name = splitext(template)[1].strip(".")
        module_path = splitext(template)[0]
        logger.info(
            f"Importing [red]🡆[/] Object [red]{class_name}[/] from [red]{module_path}[/] Module.",
            extra={"markup": True},
        )
        module = import_module(module_path)
        instance = getattr(module, class_name)

        return instance

    def __init__(self):
        """Initializes the Plan object."""
        self.config = {}  # dictionary containing patched plan definition
        self.authorized_cols = []  # authorized collaborator list
        self.cols_data_paths = {}  # collaborator data paths dict

        self.collaborator_ = None  # collaborator object
        self.aggregator_ = None  # aggregator object
        self.assigner_ = None  # assigner object
        self.connector_ = None  # OpenFL Connector object

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
        self.hash_ = sha384(dump(self.config).encode("utf-8"))
        logger.info(
            f"FL-Plan hash is [blue]{self.hash_.hexdigest()}[/]",
            extra={"markup": True},
        )

        return self.hash_.hexdigest()

    def resolve(self):
        """Resolve the federation settings."""
        self.federation_uuid = f"{self.name}_{self.hash[:8]}"
        self.aggregator_uuid = f"aggregator_{self.federation_uuid}"

        self.rounds_to_train = self.config["aggregator"][SETTINGS]["rounds_to_train"]

        if self.config["network"][SETTINGS]["agg_addr"] == AUTO:
            self.config["network"][SETTINGS]["agg_addr"] = getfqdn_env()

        if self.config["network"][SETTINGS]["agg_port"] == AUTO:
            self.config["network"][SETTINGS]["agg_port"] = (
                int(self.hash[:8], 16) % (60999 - 49152) + 49152
            )

    def get_assigner(self):
        """Get the plan task assigner."""
        defaults = self.config.get(
            "assigner",
            {TEMPLATE: "openfl.component.Assigner", SETTINGS: {}},
        )

        defaults[SETTINGS]["authorized_cols"] = self.authorized_cols
        defaults[SETTINGS]["rounds_to_train"] = self.rounds_to_train
        defaults[SETTINGS]["tasks"] = self.get_tasks()

        if self.assigner_ is None:
            self.assigner_ = Plan.build(**defaults)

        return self.assigner_

    def get_connector(self):
        """Get OpenFL Connector object."""
        defaults = self.config.get("connector")
        logger.info("Connector defaults: %s", defaults)

        if self.connector_ is None and defaults:
            self.connector_ = Plan.build(**defaults)

        return self.connector_

    def get_tasks(self):
        """Get federation tasks."""
        tasks = self.config.get("tasks", {})
        tasks.pop(DEFAULTS, None)
        tasks.pop(SETTINGS, None)
        for task in tasks:
            aggregation_type = tasks[task].get("aggregation_type")
            if aggregation_type is None:
                aggregation_type = WeightedAverage()
            elif isinstance(aggregation_type, dict):
                if SETTINGS not in aggregation_type:
                    aggregation_type[SETTINGS] = {}
                aggregation_type = Plan.build(**aggregation_type)
                if not isinstance(aggregation_type, AggregationFunction):
                    raise NotImplementedError(
                        f"""{task} task aggregation type does not implement an interface:
    openfl.interface.aggregation_functions.AggregationFunction
    """
                    )
            tasks[task]["aggregation_type"] = aggregation_type
        return tasks

    def get_aggregator(self, tensor_dict=None):
        """Get federation aggregator.

        This method retrieves the federation aggregator. If the aggregator
        does not exist, it is built using the configuration settings and the
        provided tensor dictionary.

        Args:
            tensor_dict (dict, optional): The initial tensor dictionary to use
                when building the aggregator. Defaults to None.

        Returns:
            self.aggregator_ (Aggregator): The federation aggregator.

        Raises:
            TypeError: If the log_metric_callback is not a callable object or
                cannot be imported from code.
        """
        defaults = self.config.get(
            "aggregator",
            {TEMPLATE: "openfl.component.Aggregator", SETTINGS: {}},
        )

        defaults[SETTINGS]["aggregator_uuid"] = self.aggregator_uuid
        defaults[SETTINGS]["federation_uuid"] = self.federation_uuid
        defaults[SETTINGS]["authorized_cols"] = self.authorized_cols
        defaults[SETTINGS]["assigner"] = self.get_assigner()
        defaults[SETTINGS]["compression_pipeline"] = self.get_tensor_pipe()
        defaults[SETTINGS]["straggler_handling_policy"] = self.get_straggler_handling_policy()

        connector = self.get_connector()
        if connector is not None:
            defaults[SETTINGS]["connector"] = connector

        # TODO: Load callbacks from plan.

        if self.aggregator_ is None:
            self.aggregator_ = Plan.build(**defaults, initial_tensor_dict=tensor_dict)

        return self.aggregator_

    def get_tensor_pipe(self):
        """Get data tensor pipeline."""
        defaults = self.config.get(
            "compression_pipeline",
            {TEMPLATE: "openfl.pipelines.NoCompressionPipeline", SETTINGS: {}},
        )

        if self.pipe_ is None:
            self.pipe_ = Plan.build(**defaults)

        return self.pipe_

    def get_straggler_handling_policy(self):
        """Get straggler handling policy."""
        defaults = self.config.get(
            "straggler_handling_policy",
            {
                TEMPLATE: "openfl.component.aggregator.straggler_handling.WaitForAllPolicy",
                SETTINGS: {},
            },
        )

        if self.straggler_policy_ is None:
            # Prepare a partial function for the straggler policy
            self.straggler_policy_ = partial(
                Plan.import_(defaults["template"]), **defaults["settings"]
            )

        return self.straggler_policy_

    # TaskRunner API
    def get_data_loader(self, collaborator_name):
        """Get data loader for a specific collaborator.

        Args:
            collaborator_name (str): Name of the collaborator.

        Returns:
            DataLoader: Data loader for the specified collaborator.
        """
        defaults = self.config.get(
            "data_loader",
            {TEMPLATE: "openfl.federation.DataLoader", SETTINGS: {}},
        )

        defaults[SETTINGS]["data_path"] = self.cols_data_paths[collaborator_name]

        if self.loader_ is None:
            self.loader_ = Plan.build(**defaults)

        return self.loader_

    # legacy api (TaskRunner subclassing)
    def get_task_runner(self, data_loader):
        """Get task runner.

        Args:
            data_loader (DataLoader): Data loader for the tasks.

        Returns:
            TaskRunner: Task runner for the tasks.
        """
        defaults = self.config.get(
            "task_runner",
            {TEMPLATE: "openfl.federation.TaskRunner", SETTINGS: {}},
        )

        defaults[SETTINGS]["data_loader"] = data_loader

        if self.runner_ is None:
            self.runner_ = Plan.build(**defaults)

        # Define task dependencies after taskrunner has been initialized
        self.runner_.initialize_tensorkeys_for_functions()

        return self.runner_

    def get_collaborator(
        self,
        collaborator_name,
        root_certificate=None,
        private_key=None,
        certificate=None,
        task_runner=None,
        client=None,
    ):
        """Get collaborator.

        This method retrieves a collaborator. If the collaborator does not
        exist, it is built using the configuration settings and the provided
        parameters.

        Args:
            collaborator_name (str): Name of the collaborator.
            root_certificate (str, optional): Root certificate for the
                collaborator. Defaults to None.
            private_key (str, optional): Private key for the collaborator.
                Defaults to None.
            certificate (str, optional): Certificate for the collaborator.
                Defaults to None.
            task_runner (TaskRunner, optional): Task runner for the
                collaborator. Defaults to None.
            client (Client, optional): Client for the collaborator. Defaults
                to None.

        Returns:
            self.collaborator_ (Collaborator): The collaborator instance.
        """
        defaults = self.config.get(
            "collaborator",
            {TEMPLATE: "openfl.component.Collaborator", SETTINGS: {}},
        )

        defaults[SETTINGS]["collaborator_name"] = collaborator_name
        defaults[SETTINGS]["aggregator_uuid"] = self.aggregator_uuid
        defaults[SETTINGS]["federation_uuid"] = self.federation_uuid

        # TODO: Load callbacks from the plan.

        if task_runner is not None:
            defaults[SETTINGS]["task_runner"] = task_runner
        else:
            # TaskRunner subclassing API
            data_loader = self.get_data_loader(collaborator_name)
            defaults[SETTINGS]["task_runner"] = self.get_task_runner(data_loader)

        defaults[SETTINGS]["compression_pipeline"] = self.get_tensor_pipe()
        defaults[SETTINGS]["task_config"] = self.config.get("tasks", {})
        # Check if secure aggregation is enabled.
        defaults[SETTINGS]["secure_aggregation"] = (
            self.config.get("aggregator", {}).get(SETTINGS, {}).get("secure_aggregation", False)
        )
        if client is not None:
            defaults[SETTINGS]["client"] = client
        else:
            defaults[SETTINGS]["client"] = self.get_client(
                collaborator_name,
                self.aggregator_uuid,
                self.federation_uuid,
                root_certificate,
                private_key,
                certificate,
            )

        if self.collaborator_ is None:
            self.collaborator_ = Plan.build(**defaults)

        return self.collaborator_

    def get_client(
        self,
        collaborator_name,
        aggregator_uuid,
        federation_uuid,
        root_certificate=None,
        private_key=None,
        certificate=None,
    ):
        """Get gRPC client for the specified collaborator.

        Args:
            collaborator_name (str): Name of the collaborator.
            aggregator_uuid (str): UUID of the aggregator.
            federation_uuid (str): UUID of the federation.
            root_certificate (str, optional): Root certificate for the
                collaborator. Defaults to None.
            private_key (str, optional): Private key for the collaborator.
                Defaults to None.
            certificate (str, optional): Certificate for the collaborator.
                Defaults to None.

        Returns:
            AggregatorGRPCClient: gRPC client for the specified collaborator.
        """
        common_name = collaborator_name
        if not root_certificate or not private_key or not certificate:
            root_certificate = "cert/cert_chain.crt"
            certificate = f"cert/client/col_{common_name}.crt"
            private_key = f"cert/client/col_{common_name}.key"

        client_args = self.config["network"][SETTINGS]

        # patch certificates

        client_args["root_certificate"] = root_certificate
        client_args["certificate"] = certificate
        client_args["private_key"] = private_key

        client_args["aggregator_uuid"] = aggregator_uuid
        client_args["federation_uuid"] = federation_uuid
        client_args["collaborator_name"] = collaborator_name

        if self.client_ is None:
            self.client_ = AggregatorGRPCClient(**client_args)

        return self.client_

    def get_server(
        self,
        root_certificate=None,
        private_key=None,
        certificate=None,
        **kwargs,
    ):
        """Get gRPC server of the aggregator instance.

        Args:
            root_certificate (str, optional): Root certificate for the server.
                Defaults to None.
            private_key (str, optional): Private key for the server. Defaults
                to None.
            certificate (str, optional): Certificate for the server. Defaults
                to None.
            **kwargs: Additional keyword arguments.

        Returns:
            AggregatorGRPCServer: gRPC server of the aggregator instance.
        """
        common_name = self.config["network"][SETTINGS]["agg_addr"].lower()

        if not root_certificate or not private_key or not certificate:
            root_certificate = "cert/cert_chain.crt"
            certificate = f"cert/server/agg_{common_name}.crt"
            private_key = f"cert/server/agg_{common_name}.key"

        server_args = self.config["network"][SETTINGS]

        # patch certificates

        server_args.update(kwargs)
        server_args["root_certificate"] = root_certificate
        server_args["certificate"] = certificate
        server_args["private_key"] = private_key

        server_args["aggregator"] = self.get_aggregator()

        if self.server_ is None:
            self.server_ = AggregatorGRPCServer(**server_args)

        return self.server_

    def save_model_to_state_file(self, tensor_dict, round_number, output_path):
        """Save model weights to a protobuf state file.

        This method serializes the model weights into a protobuf format and saves
        them to a file. The serialization is done using the tensor pipe to ensure
        proper compression and formatting.

        Args:
            tensor_dict (dict): Dictionary containing model weights and their
                corresponding tensors.
            round_number (int): The current federation round number.
            output_path (str): Path where the serialized model state will be
                saved.

        Raises:
            Exception: If there is an error during model proto creation or saving
                to file.
        """
        from openfl.protocols import utils  # Import here to avoid circular imports

        # Get tensor pipe to properly serialize the weights
        tensor_pipe = self.get_tensor_pipe()

        # Create and save the protobuf message
        try:
            model_proto = utils.construct_model_proto(
                tensor_dict=tensor_dict, round_number=round_number, tensor_pipe=tensor_pipe
            )
            utils.dump_proto(model_proto=model_proto, fpath=output_path)
        except Exception as e:
            logger.error(f"Failed to create or save model proto: {e}")
            raise

    def verify(self):
        """
        This function checks for inconsistencies in the plan config, for example, checks if two
        non-compatible features are enabled at the same time.
        """
        if self.config["aggregator"][SETTINGS].get("secure_aggregation"):
            # TODO: Secure aggregation requires all collaborators to participate in all rounds and
            # can not tolerate dropouts. Hence, if `secure_aggregation: true` in aggregator, there
            # should be no `straggler_handling_policy` defined or use `WaitForAllPolicy` (default
            # straggler handling policy).
            straggler_handling_policy = self.config.get(
                "straggler_handling_policy",
                {
                    TEMPLATE: "openfl.component.aggregator.straggler_handling.WaitForAllPolicy",
                    SETTINGS: {},
                },
            )
            if straggler_handling_policy.get(TEMPLATE) not in [
                "openfl.component.WaitForAllPolicy",
                "openfl.component.aggregator.WaitForAllPolicy",
                "openfl.component.aggregator.straggler_handling.WaitForAllPolicy",
            ]:
                raise Exception(
                    "Only WaitForAllPolicy straggler handling is supported with secure aggregation."
                )
