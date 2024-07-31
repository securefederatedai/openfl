# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""PKI CLI."""

import logging
import os
import sys
from pathlib import Path

from click import Path as ClickPath
from click import group, option, pass_context, password_option

from openfl.utilities.ca.ca import (
    CA_CONFIG_JSON,
    CA_PASSWORD_FILE,
    CA_PKI_DIR,
    CA_STEP_CONFIG_DIR,
    certify,
    get_ca_bin_paths,
    get_token,
    install,
    remove_ca,
    run_ca,
)

logger = logging.getLogger(__name__)

CA_URL = "localhost:9123"


@group()
@pass_context
def pki(context):
    """Manage Step-ca PKI.

    Args:
        context (click.core.Context): Click context.
    """
    context.obj["group"] = "pki"


@pki.command(name="run")
@option("-p", "--ca-path", required=True, help="The ca path", type=ClickPath())
def run_(ca_path):
    """Run CA server.

    Args:
        ca_path (str): The ca path.
    """
    run(ca_path)


def run(ca_path):
    """Run CA server.

    Args:
        ca_path (str): The ca path.
    """
    ca_path = Path(ca_path).absolute()
    step_config_dir = ca_path / CA_STEP_CONFIG_DIR
    pki_dir = ca_path / CA_PKI_DIR
    password_file = pki_dir / CA_PASSWORD_FILE
    ca_json = step_config_dir / CA_CONFIG_JSON
    _, step_ca_path = get_ca_bin_paths(ca_path)
    if (
        not os.path.exists(step_config_dir)
        or not os.path.exists(pki_dir)
        or not os.path.exists(password_file)
        or not os.path.exists(ca_json)
        or not os.path.exists(step_ca_path)
    ):
        logger.error("CA is not installed or corrupted, please install it first")
        sys.exit(1)
    run_ca(step_ca_path, password_file, ca_json)


@pki.command(name="install")
@option("-p", "--ca-path", required=True, help="The ca path", type=ClickPath())
@password_option(prompt="The password will encrypt some ca files \nEnter the password")
@option("--ca-url", required=False, default=CA_URL)
def install_(ca_path, password, ca_url):
    """Create a ca workspace.

    Args:
        ca_path (str): The ca path.
        password (str): The password will encrypt some ca files.
        ca_url (str): CA URL.
    """
    ca_path = Path(ca_path).absolute()
    install(ca_path, ca_url, password)


@pki.command(name="uninstall")
@option("-p", "--ca-path", required=True, help="The CA path", type=ClickPath())
def uninstall(ca_path):
    """Remove step-CA.

    Args:
        ca_path (str): The CA path.
    """
    ca_path = Path(ca_path).absolute()
    remove_ca(ca_path)


@pki.command(name="get-token")
@option("-n", "--name", required=True)
@option("--ca-url", required=False, default=CA_URL)
@option(
    "-p",
    "--ca-path",
    default=".",
    help="The CA path",
    type=ClickPath(exists=True),
)
def get_token_(name, ca_url, ca_path):
    """Create authentication token.

    Args:
        name (str): Common name for following certificate (aggregator fqdn or
            collaborator name).
        ca_url (str): Full URL of CA server.
        ca_path (str): The path to CA binaries.
    """
    ca_path = Path(ca_path).absolute()
    token = get_token(name, ca_url, ca_path)
    print("Token:")
    print(token)


@pki.command(name="certify")
@option("-n", "--name", required=True)
@option("-t", "--token", "token_with_cert", required=True)
@option(
    "-c",
    "--certs-path",
    required=False,
    default=Path(".") / "cert",
    help="The path where certificates will be stored",
    type=ClickPath(),
)
@option(
    "-p",
    "--ca-path",
    default=".",
    help="The path to CA client",
    type=ClickPath(exists=True),
    required=False,
)
def certify_(name, token_with_cert, certs_path, ca_path):
    """Create an envoy workspace.

    Args:
        name (str): Common name for following certificate (aggregator fqdn or
            collaborator name).
        token_with_cert (str): Authentication token.
        certs_path (str): The path where certificates will be stored.
        ca_path (str): The path to CA client.
    """
    certs_path = Path(certs_path).absolute()
    ca_path = Path(ca_path).absolute()
    certs_path.mkdir(parents=True, exist_ok=True)
    certify(name, certs_path, token_with_cert, ca_path)
