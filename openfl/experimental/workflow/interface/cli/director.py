# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Director CLI."""

import logging
import sys
from pathlib import Path

import click
from click import Path as ClickPath
from click import group, option, pass_context
from dynaconf import Validator

from openfl.experimental.workflow.component.director import Director
from openfl.experimental.workflow.transport import DirectorGRPCServer
from openfl.utilities import merge_configs
from openfl.utilities.path_check import is_directory_traversal

logger = logging.getLogger(__name__)


@group()
@pass_context
def director(context):
    """Manage Federated Learning Director.

    Args:
        context (click.core.Context): Click context.
    """
    context.obj["group"] = "director"


@director.command(name="start")
@option(
    "-c",
    "--director-config-path",
    default="director.yaml",
    help="The director config file path",
    type=ClickPath(exists=True),
)
@option(
    "--tls/--disable-tls",
    default=True,
    is_flag=True,
    help="Use TLS or not (By default TLS is enabled)",
)
@option(
    "-rc",
    "--root-cert-path",
    "root_certificate",
    required=False,
    type=ClickPath(exists=True),
    default=None,
    help="Path to a root CA cert",
)
@option(
    "-pk",
    "--private-key-path",
    "private_key",
    required=False,
    type=ClickPath(exists=True),
    default=None,
    help="Path to a private key",
)
@option(
    "-oc",
    "--public-cert-path",
    "certificate",
    required=False,
    type=ClickPath(exists=True),
    default=None,
    help="Path to a signed certificate",
)
def start(director_config_path, tls, root_certificate, private_key, certificate):
    """Start the director service.

    Args:
        director_config_path (str): The director config file path.
        tls (bool): Use TLS or not.
        root_certificate (str): Path to a root CA cert.
        private_key (str): Path to a private key.
        certificate (str): Path to a signed certificate.
    """

    director_config_path = Path(director_config_path).absolute()
    logger.info("🧿 Starting the Director Service.")
    if is_directory_traversal(director_config_path):
        click.echo("The director config file path is out of the openfl workspace scope.")
        sys.exit(1)
    config = merge_configs(
        settings_files=director_config_path,
        overwrite_dict={
            "root_certificate": root_certificate,
            "private_key": private_key,
            "certificate": certificate,
        },
        validators=[
            Validator("settings.listen_host", default="localhost"),
            Validator("settings.listen_port", default=50051, gte=1024, lte=65535),
            Validator("settings.install_requirements", default=True),
            Validator(
                "settings.envoy_health_check_period",
                default=60,  # in seconds
                gte=1,
                lte=24 * 60 * 60,
            ),
        ],
    )

    if config.root_certificate:
        config.root_certificate = Path(config.root_certificate).absolute()

    if config.private_key:
        config.private_key = Path(config.private_key).absolute()

    if config.certificate:
        config.certificate = Path(config.certificate).absolute()

    director_server = DirectorGRPCServer(
        director_cls=Director,
        tls=tls,
        root_certificate=config.root_certificate,
        private_key=config.private_key,
        certificate=config.certificate,
        listen_host=config.settings.listen_host,
        listen_port=config.settings.listen_port,
        envoy_health_check_period=config.settings.envoy_health_check_period,
        install_requirements=config.settings.install_requirements,
        director_config=director_config_path,
    )
    director_server.start()
