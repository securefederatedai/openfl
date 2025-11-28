# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""PKI CLI."""

import logging
import os
import sys
from pathlib import Path

from click import Path as ClickPath
from click import echo, group, option, pass_context, password_option

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
    """Manage Step-CA PKI."""
    context.obj["group"] = "pki"


@pki.command(name="run")
@option("-p", "--ca-path", required=True, help="The ca path", type=ClickPath())
def run_(ca_path):
    """Starts a CA server."""
    run(ca_path)


def run(ca_path):
    """Starts a CA server.

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
@option("-p", "--ca-path", required=True, help="Path to CA.", type=ClickPath())
@password_option(prompt="The password will encrypt CA files. \nEnter the password: ")
@option("--ca-url", required=False, default=CA_URL, show_default=True)
def install_(ca_path, password, ca_url):
    """Creates a CA workspace, optionally password protected."""
    ca_path = Path(ca_path).absolute()
    install(ca_path, ca_url, password)


@pki.command(name="uninstall")
@option("-p", "--ca-path", required=True, help="Path to CA to be uninstalled.", type=ClickPath())
def uninstall(ca_path):
    """Removes Step-CA."""
    ca_path = Path(ca_path).absolute()
    remove_ca(ca_path)


@pki.command(name="get-token")
@option("-n", "--name", required=True)
@option(
    "--ca-url", required=False, default=CA_URL, help="Full URL of CA server.", show_default=True
)
@option(
    "-p",
    "--ca-path",
    default=".",
    help="Path to CA binaries, defaults to current directory.",
    type=ClickPath(exists=True),
)
def get_token_(name, ca_url, ca_path):
    """Creates an authentication token."""
    ca_path = Path(ca_path).absolute()
    token = get_token(name, ca_url, ca_path)
    echo(f"Token: {token}")


@pki.command(name="certify")
@option(
    "-n",
    "--name",
    required=True,
    help=(
        "Subject Alternative Name (SAN) to use for certificate. "
        "Use FQDN for aggregator, and common name for collaborator"
    ),
)
@option("-t", "--token", "token_with_cert", required=True, help="Authentication token.")
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
    help="Path to CA client, defaults to current directory.",
    type=ClickPath(exists=True),
    required=False,
    show_default=True,
)
def certify_(name, token_with_cert, certs_path, ca_path):
    """Generates a certificate for the given name."""
    certs_path = Path(certs_path).absolute()
    ca_path = Path(ca_path).absolute()
    certs_path.mkdir(parents=True, exist_ok=True)
    certify(name, certs_path, token_with_cert, ca_path)
