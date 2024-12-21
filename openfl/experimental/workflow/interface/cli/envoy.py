# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Envoy CLI."""

import logging
import sys
from pathlib import Path

import click
from click import Path as ClickPath
from click import group, option, pass_context
from dynaconf import Validator

from openfl.experimental.workflow.component.envoy import Envoy
from openfl.utilities import click_types, merge_configs
from openfl.utilities.path_check import is_directory_traversal

logger = logging.getLogger(__name__)


@group()
@pass_context
def envoy(context):
    """Manage Federated Learning Envoy.

    Args:
        context (click.core.Context): Click context.
    """
    context.obj["group"] = "envoy"


@envoy.command(name="start")
@option("-n", "--envoy_name", required=True, help="Current shard name")
@option(
    "-dh",
    "--director-host",
    required=True,
    help="The FQDN of the federation director",
    type=click_types.FQDN,
)
@option(
    "-dp",
    "--director-port",
    required=True,
    help="The federation director port",
    type=click.IntRange(1, 65535),
)
@option(
    "--tls/--disable-tls",
    default=True,
    is_flag=True,
    help="Use TLS or not (By default TLS is enabled)",
)
@option(
    "-ec",
    "--envoy-config-path",
    default="envoy_config.yaml",
    help="The envoy config path",
    type=ClickPath(exists=True),
)
@option(
    "-rc",
    "--root-cert-path",
    "root_certificate",
    default=None,
    help="Path to a root CA cert",
    type=ClickPath(exists=True),
)
@option(
    "-pk",
    "--private-key-path",
    "private_key",
    default=None,
    help="Path to a private key",
    type=ClickPath(exists=True),
)
@option(
    "-oc",
    "--public-cert-path",
    "certificate",
    default=None,
    help="Path to a signed certificate",
    type=ClickPath(exists=True),
)
def start_(
    envoy_name,
    director_host,
    director_port,
    tls,
    envoy_config_path,
    root_certificate,
    private_key,
    certificate,
):
    """Start the Envoy.

    Args:
        envoy_name (str): Name of the Envoy.
        director_host (str): The FQDN of the federation director.
        director_port (int): The federation director port.
        tls (bool): Use TLS or not.
        envoy_config_path (str): The envoy config path.
        root_certificate (str): Path to a root CA cert.
        private_key (str): Path to a private key.
        certificate (str): Path to a signed certificate.
    """

    logger.info("🧿 Starting the Envoy.")
    if is_directory_traversal(envoy_config_path):
        click.echo("The envoy config path is out of the openfl workspace scope.")
        sys.exit(1)

    config = merge_configs(
        settings_files=envoy_config_path,
        overwrite_dict={
            "root_certificate": root_certificate,
            "private_key": private_key,
            "certificate": certificate,
        },
        validators=[
            Validator("params.install_requirements", default=True),
        ],
    )

    # Parse envoy parameters
    envoy_params = config.get("params", {})
    if envoy_params:
        install_requirements = envoy_params["install_requirements"]
    else:
        install_requirements = False

    if config.root_certificate:
        config.root_certificate = Path(config.root_certificate).absolute()
    if config.private_key:
        config.private_key = Path(config.private_key).absolute()
    if config.certificate:
        config.certificate = Path(config.certificate).absolute()

    envoy = Envoy(
        envoy_name=envoy_name,
        director_host=director_host,
        director_port=director_port,
        envoy_config=Path(envoy_config_path).absolute(),
        root_certificate=config.root_certificate,
        private_key=config.private_key,
        certificate=config.certificate,
        tls=tls,
        install_requirements=install_requirements,
    )

    envoy.start()
