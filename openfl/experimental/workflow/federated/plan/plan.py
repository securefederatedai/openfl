# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Plan module."""

import inspect
from hashlib import sha384
from importlib import import_module, reload
from logging import getLogger
from os.path import splitext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from yaml import SafeDumper, dump, safe_load

from openfl.experimental.workflow.transport import AggregatorGRPCClient, AggregatorGRPCServer
from openfl.utilities.utils import getfqdn_env

if TYPE_CHECKING:
    from openfl.experimental.workflow.component import Aggregator, Collaborator
    from openfl.experimental.workflow.interface import FLSpec

SETTINGS = "settings"
TEMPLATE = "template"
AUTO = "auto"


class Plan:
    """A class used to represent a Federated Learning plan.

    This class provides methods to manage and manipulate federated learning
    plans.

    Attributes:
        logger (Logger): Logger instance for the class.
        config (dict): Dictionary containing patched plan definition.
        authorized_cols (list): Authorized collaborator list.
        cols_data_paths (dict): Collaborator data paths dictionary.
        collaborator_ (Collaborator): Collaborator object.
        aggregator_ (Aggregator): Aggregator object.
        server_ (AggregatorGRPCServer): gRPC server object.
        client_ (AggregatorGRPCClient): gRPC client object.
        hash_ (str): Hash of the instance.
    """

    logger = getLogger(__name__)

    @staticmethod
    def load(yaml_path: Path, default: dict = None) -> dict:
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
    def dump(yaml_path, config, freeze=False) -> None:
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
                Plan.logger.info(f"{yaml_path.name} is already frozen")
                return
            frozen_yaml_path.write_text(dump(config))
            frozen_yaml_path.chmod(0o400)
            Plan.logger.info(f"{yaml_path.name} frozen successfully")
        else:
            yaml_path.write_text(dump(config))

    @staticmethod
    def parse(
        plan_config_path: Path,
        resolve=True,
    ) -> "Plan":
        """Parse the Federated Learning plan.

        Args:
            plan_config_path (string): The filepath to the federated learning
                                       plan
        Returns:
            A federated learning plan object
        """
        try:
            plan = Plan()
            plan.config = Plan.load(plan_config_path)  # load plan configuration
            plan.name = plan_config_path.name
            plan.files = [plan_config_path]  # collect all the plan files

            # ensure 'settings' appears in each top-level section
            for section in plan.config.keys():
                if plan.config[section].get(SETTINGS) is None:
                    plan.config[section][SETTINGS] = {}

            if resolve:
                plan.resolve()

                Plan.logger.info(
                    f"Parsing Federated Learning Plan : [green]SUCCESS[/] : "
                    f"[blue]{plan_config_path}[/].",
                    extra={"markup": True},
                )
                Plan.dump(plan_config_path, plan.config)
                Plan.logger.info(dump(plan.config))

            return plan

        except Exception:
            Plan.logger.exception(
                f"Parsing Federated Learning Plan : [red]FAILURE[/] : [blue]{plan_config_path}[/].",
                extra={"markup": True},
            )
            raise

    @staticmethod
    def accept_args(cls) -> bool:
        """Determines whether a class's constructor (__init__ method) accepts
        variable positional arguments (*args).

        Returns:
            Boolean: True or False
        """
        init_signature = inspect.signature(cls.__init__)
        for param in init_signature.parameters.values():
            if param.kind == param.VAR_POSITIONAL:
                return True
        return False

    @staticmethod
    def build(template, settings, **override) -> object:
        """Create an instance of a openfl Component or Federated
        DataLoader/TaskRunner.

        Args:
            template: Fully qualified class template path
            settings: Keyword arguments to class constructor

        Returns:
            A Python object
        """
        class_name = splitext(template)[1].strip(".")
        module_path = splitext(template)[0]

        Plan.logger.info(f"Building `{template}` Module.")
        Plan.logger.debug(f"Settings {settings}")
        Plan.logger.debug(f"Override {override}")

        settings.update(**override)
        module = import_module(module_path)
        # Reload the module to ensure the flow is rebuilt with updated changes
        module = reload(module)

        if Plan.accept_args(getattr(module, class_name)):
            args = list(settings.values())
            instance = getattr(module, class_name)(*args)
        else:
            instance = getattr(module, class_name)(**settings)

        return instance

    @staticmethod
    def import_(template) -> object:
        """Import an instance of a openfl Component or Federated
        DataLoader/TaskRunner.

        Args:
            template: Fully qualified object path

        Returns:
            A Python object
        """
        class_name = splitext(template)[1].strip(".")
        module_path = splitext(template)[0]
        Plan.logger.info(
            f"Importing [red]🡆[/] Object [red]{class_name}[/] from [red]{module_path}[/] Module.",
            extra={"markup": True},
        )
        module = import_module(module_path)
        # FIXME: Reload module to refresh private attributes; optimize this.
        module = reload(module)
        instance = getattr(module, class_name)

        return instance

    def __init__(self) -> None:
        """Initialize."""
        self.config = {}  # dictionary containing patched plan definition
        self.authorized_cols = []  # authorized collaborator list

        self.collaborator_ = None  # collaborator object
        self.aggregator_ = None  # aggregator object

        self.server_ = None  # gRPC server object
        self.client_ = None  # gRPC client object

        self.hash_ = None

    @property
    def hash(self) -> str:  # NOQA
        """Generate hash for this instance."""
        self.hash_ = sha384(dump(self.config).encode("utf-8"))
        Plan.logger.info(
            f"FL-Plan hash is [blue]{self.hash_.hexdigest()}[/]",
            extra={"markup": True},
        )

        return self.hash_.hexdigest()

    def resolve(self) -> None:
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

    def get_aggregator(self, director_config) -> "Aggregator":
        """Get federation aggregator.

        Args:
            director_config: Path to director config file.

        Returns:
            self.aggregator_ (Aggregator): The federation aggregator.
        """
        defaults = self.config.get(
            "aggregator",
            {TEMPLATE: "openfl.experimental.workflow.Aggregator", SETTINGS: {}},
        )

        defaults[SETTINGS]["aggregator_uuid"] = self.aggregator_uuid
        defaults[SETTINGS]["federation_uuid"] = self.federation_uuid
        defaults[SETTINGS]["authorized_cols"] = self.authorized_cols

        private_attrs_callable, private_attrs_kwargs, private_attributes = self.get_private_attr(
            "aggregator", director_config
        )
        defaults[SETTINGS]["private_attributes_callable"] = private_attrs_callable
        defaults[SETTINGS]["private_attributes_kwargs"] = private_attrs_kwargs
        defaults[SETTINGS]["private_attributes"] = private_attributes

        defaults[SETTINGS]["flow"] = self.get_flow()
        defaults[SETTINGS]["checkpoint"] = defaults[SETTINGS]["flow"].checkpoint

        log_metric_callback = defaults[SETTINGS].get("log_metric_callback")
        if log_metric_callback:
            if isinstance(log_metric_callback, dict):
                log_metric_callback = Plan.import_(**log_metric_callback)
            elif not callable(log_metric_callback):
                raise TypeError(
                    f"log_metric_callback should be callable object "
                    f"or be import from code part, get {log_metric_callback}"
                )
        defaults[SETTINGS]["log_metric_callback"] = log_metric_callback

        if self.aggregator_ is None:
            self.aggregator_ = Plan.build(**defaults)

        return self.aggregator_

    def get_collaborator(
        self,
        collaborator_name,
        envoy_config,
        root_certificate=None,
        private_key=None,
        certificate=None,
        client=None,
        tls=True,
    ) -> "Collaborator":
        """Get collaborator.

        This method retrieves a collaborator. If the collaborator does not
        exist, it is built using the configuration settings and the provided
        parameters.

        Args:
            collaborator_name (str): Name of the collaborator.
            envoy_config (Path): Path to envoy_config.yaml.
            root_certificate (str, optional): Root certificate for the
                collaborator. Defaults to None.
            private_key (str, optional): Private key for the collaborator.
                Defaults to None.
            certificate (str, optional): Certificate for the collaborator.
                Defaults to None.
            client (Client, optional): Client for the collaborator. Defaults
                to None.
            tls (bool): Whether to use TLS for the connection.

        Returns:
            self.collaborator_ (Collaborator): The collaborator instance.
        """
        defaults = self.config.get(
            "collaborator",
            {TEMPLATE: "openfl.experimental.workflow.Collaborator", SETTINGS: {}},
        )

        defaults[SETTINGS]["collaborator_name"] = collaborator_name
        defaults[SETTINGS]["aggregator_uuid"] = self.aggregator_uuid
        defaults[SETTINGS]["federation_uuid"] = self.federation_uuid

        private_attrs_callable, private_attrs_kwargs, private_attributes = self.get_private_attr(
            collaborator_name, envoy_config
        )
        defaults[SETTINGS]["private_attributes_callable"] = private_attrs_callable
        defaults[SETTINGS]["private_attributes_kwargs"] = private_attrs_kwargs
        defaults[SETTINGS]["private_attributes"] = private_attributes

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
                tls,
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
        tls=True,
    ) -> AggregatorGRPCClient:
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
            tls (bool): Whether to use TLS for the connection.

        Returns:
            AggregatorGRPCClient: gRPC client for the specified collaborator.
        """
        client_args = self.config["network"][SETTINGS]

        # patch certificates

        client_args["root_certificate"] = root_certificate
        client_args["certificate"] = certificate
        client_args["private_key"] = private_key

        client_args["aggregator_uuid"] = aggregator_uuid
        client_args["federation_uuid"] = federation_uuid

        if self.client_ is None:
            self.client_ = AggregatorGRPCClient(**client_args)

        return self.client_

    def get_server(
        self,
        director_config,
        root_certificate=None,
        private_key=None,
        certificate=None,
        tls=True,
        **kwargs,
    ) -> AggregatorGRPCServer:
        """Get gRPC server of the aggregator instance.

        Args:
            root_certificate (str, optional): Root certificate for the server.
                Defaults to None.
            director_config (Path): Path to director_config.yaml.
            private_key (str, optional): Private key for the server. Defaults
                to None.
            certificate (str, optional): Certificate for the server. Defaults
                to None.
            tls (bool): Whether to use TLS for the connection.
            **kwargs: Additional keyword arguments.

        Returns:
            AggregatorGRPCServer: gRPC server of the aggregator instance.
        """
        server_args = self.config["network"][SETTINGS]

        # patch certificates

        server_args.update(kwargs)
        server_args["root_certificate"] = root_certificate
        server_args["certificate"] = certificate
        server_args["private_key"] = private_key
        server_args["tls"] = tls

        server_args["aggregator"] = self.get_aggregator(director_config)

        if self.server_ is None:
            self.server_ = AggregatorGRPCServer(**server_args)

        return self.server_

    def get_flow(self) -> "FLSpec":
        """Instantiates federated flow object.

        Returns:
            flow_: FLSpec instance
        """
        defaults = self.config.get(
            "federated_flow",
            {TEMPLATE: self.config["federated_flow"]["template"], SETTINGS: {}},
        )
        defaults = self.import_kwargs_modules(defaults)

        self.flow_ = Plan.build(**defaults)
        return self.flow_

    def import_kwargs_modules(self, defaults) -> Dict[str, Any]:
        """
        Imports and resolves class references in a nested settings structure.
        Args:
            defaults (Dict[str, Any]): A dictionary of settings, containing module paths
                and class names as strings in its nested structure.

        Returns:
            Dict[str, Any]: The updated settings dictionary with resolved classes and attributes
                from the imported modules.
        """

        def import_nested_settings(settings):
            for key, value in settings.items():
                if isinstance(value, dict):
                    settings[key] = import_nested_settings(value)
                elif isinstance(value, str):
                    class_name = splitext(value)[1].strip(".")
                    if class_name:
                        module_path = splitext(value)[0]
                        try:
                            if import_module(module_path):
                                module = import_module(module_path)
                                value_defaults_data = {
                                    "template": value,
                                    "settings": settings.get("settings", {}),
                                }
                                attr = getattr(module, class_name)

                                if not inspect.isclass(attr):
                                    settings[key] = attr
                                else:
                                    settings = Plan.build(**value_defaults_data)
                        except ImportError:
                            raise ImportError(f"Cannot import {value}.")
            return settings

        defaults[SETTINGS] = import_nested_settings(defaults[SETTINGS])
        return defaults

    def get_private_attr(
        self, private_attr_name, config
    ) -> Tuple[Optional[dict], Optional[dict], dict]:
        """
        Retrieves private attributes defined in a configuration or data file.

        Args:
            private_attr_name (str): The name of the participant (Aggregator or Collaborator)
                whose private attribute is to be retrieved.
            config (Path): Path to the config file.

        Returns:
            Tuple: A tuple containing:
                - private_attrs_callable (Optional[dict]): A dictionary containing callable
                    function information, or None if not applicable.
                - private_attrs_kwargs (Optional[dict]): A dictionary of arguments for the
                    callable function, or None if not applicable.
                - private_attributes (dict): A dictionary of private attributes,
                    or an empty dictionary if none are found.
        """
        private_attrs_callable = private_attrs_kwargs = None
        private_attributes = {}

        d = Plan.load(config)

        if d and d.get(private_attr_name, None):
            callable_func = d.get(private_attr_name, {}).get("callable_func")
            private_attributes = d.get(private_attr_name, {}).get("private_attributes")
            if callable_func and private_attributes:
                logger = getLogger(__name__)
                logger.warning(
                    f"Warning: {private_attr_name} private attributes "
                    "will be initialized via callable and "
                    "attributes directly specified "
                    "will be ignored"
                )

            if callable_func is not None:
                private_attrs_callable = {
                    "template": d.get(private_attr_name)["callable_func"]["template"]
                }

                private_attrs_kwargs = self.import_kwargs_modules(
                    d.get(private_attr_name)["callable_func"]
                )["settings"]

                if isinstance(private_attrs_callable, dict):
                    private_attrs_callable = Plan.import_(**private_attrs_callable)
            elif private_attributes:
                private_attributes = Plan.import_(d.get(private_attr_name)["private_attributes"])
            elif not callable(private_attrs_callable):
                raise TypeError(
                    f"private_attrs_callable should be callable object "
                    f"or be import from code part, get {private_attrs_callable}"
                )

        return (
            private_attrs_callable,
            private_attrs_kwargs,
            private_attributes,
        )
