# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Collaborator module."""

import os
import sys
from glob import glob
from logging import getLogger
from os import remove
from os.path import basename, isfile, join, splitext
from pathlib import Path
from shutil import copy, copytree, ignore_patterns, make_archive, unpack_archive
from tempfile import mkdtemp

from click import Path as ClickPath
from click import confirm, echo, group, option, pass_context, prompt, style
from yaml import FullLoader, dump, load

from openfl.cryptography.ca import sign_certificate
from openfl.cryptography.io import get_csr_hash, read_crt, read_csr, read_key, write_crt, write_key
from openfl.cryptography.participant import generate_csr
from openfl.federated import Plan
from openfl.federated.data.sources.data_sources_json_parser import DataSourcesJsonParser
from openfl.interface.cli_helper import CERT_DIR
from openfl.utilities.path_check import is_directory_traversal
from openfl.utilities.utils import rmtree

logger = getLogger(__name__)


@group()
@pass_context
def collaborator(context):
    """Manage Federated Learning Collaborator."""
    context.obj["group"] = "service"


@collaborator.command(name="start")
@option(
    "-p",
    "--plan",
    required=False,
    help="Path to an FL plan.",
    default="plan/plan.yaml",
    type=ClickPath(exists=True),
    show_default=True,
)
@option(
    "-d",
    "--data_config",
    required=False,
    help="The dataset shard configuration file.",
    default="plan/data.yaml",
    type=ClickPath(exists=True),
    show_default=True,
)
@option(
    "-n",
    "--collaborator_name",
    required=True,
    help="The certified common name of the collaborator.",
)
def start_(plan, collaborator_name, data_config):
    """Starts a collaborator service."""
    if plan and is_directory_traversal(plan):
        echo("Federated learning plan path is out of the openfl workspace scope.")
        sys.exit(1)
    if data_config and is_directory_traversal(data_config):
        echo("The data set/shard configuration file path is out of the openfl workspace scope.")
        sys.exit(1)

    plan_obj = Plan.parse(
        plan_config_path=Path(plan).absolute(),
        data_config_path=Path(data_config).absolute(),
    )

    # TODO: Need to restructure data loader config file loader
    logger.info(f"Data paths: {plan_obj.cols_data_paths}")
    echo(f"Data = {plan_obj.cols_data_paths}")
    logger.info("🧿 Starting a Collaborator Service.")

    collaborator = plan_obj.get_collaborator(collaborator_name)
    collaborator.run()


@collaborator.command(name="ping")
@option(
    "-p",
    "--plan",
    required=False,
    help="Path to an FL plan.",
    default="plan/plan.yaml",
    type=ClickPath(exists=True),
    show_default=True,
)
@option(
    "-d",
    "--data_config",
    required=False,
    help="The dataset shard configuration file.",
    default="plan/data.yaml",
    type=ClickPath(exists=True),
    show_default=True,
)
@option(
    "-n",
    "--collaborator_name",
    required=True,
    help="The certified common name of the collaborator.",
)
def ping_(plan, collaborator_name, data_config):
    """Ping the aggregator without starting any tasks."""

    if plan and is_directory_traversal(plan):
        echo("Federated learning plan path is out of the openfl workspace scope.")
        sys.exit(1)
    if data_config and is_directory_traversal(data_config):
        echo("The data set/shard configuration file path is out of the openfl workspace scope.")
        sys.exit(1)

    fl_plan = Plan.parse(
        plan_config_path=Path(plan).absolute(),
        data_config_path=Path(data_config).absolute(),
    )

    agg_addr = fl_plan.config["network"]["settings"]["agg_addr"]
    agg_port = fl_plan.config["network"]["settings"]["agg_port"]
    use_tls = fl_plan.config["network"]["settings"]["use_tls"]
    protocol = "TLS" if use_tls else "TCP"

    logger.info(
        f"🧿 Testing connectivity with the Aggregator at {agg_addr}:{agg_port} via {protocol}..."
    )
    fl_plan.get_collaborator(collaborator_name).ping()

    logger.info(f"The Aggregator is reachable at {agg_addr}:{agg_port}")
    logger.info(f"{protocol} connection established.")


@collaborator.command(name="create")
@option(
    "-n",
    "--collaborator_name",
    required=True,
    help="The certified common name of the collaborator.",
)
@option(
    "-d",
    "--data_path",
    help="The data path to be associated with the collaborator.",
)
@option("-s", "--silent", help="If set, skips manual confirmation.", is_flag=True)
def create_(collaborator_name, data_path, silent):
    """Registers given dataset path in the ``plan/data.yaml`` file."""
    create(collaborator_name, data_path, silent)


def create(collaborator_name, data_path, silent):
    """Creates a user for an experiment.

    Args:
        collaborator_name (str): The certified common name of the collaborator.
        data_path (str): The data path to be associated with the collaborator.
        silent (bool): Do not prompt.
    """
    if data_path and is_directory_traversal(data_path):
        echo("Data path is out of the openfl workspace scope.")
        sys.exit(1)

    common_name = f"{collaborator_name}".lower()

    # TODO: There should be some association with the plan made here as well
    register_data_path(common_name, data_path=data_path, silent=silent)


def register_data_path(collaborator_name, data_path=None, silent=False):
    """Register dataset path in the plan/data.yaml file.

    Args:
        collaborator_name (str): The collaborator whose data path to be
            defined.
        data_path (str, optional): Data path. Defaults to None.
        silent (bool, optional): Silent operation (don't prompt). Defaults to
            False.
    """

    if data_path and is_directory_traversal(data_path):
        echo("Data path is out of the openfl workspace scope.")
        sys.exit(1)

    # Ask for the data directory
    default_data_path = f"data/{collaborator_name}"
    if not silent and data_path is None:
        dir_path = prompt(
            "\nWhere is the data (or what is the rank)"
            " for collaborator " + style(f"{collaborator_name}", fg="green") + " ? ",
            default=default_data_path,
        )
    elif data_path is not None:
        dir_path = data_path
    else:
        # TODO: Need to figure out the default for this.
        dir_path = default_data_path

    # Read the data.yaml file
    d = {}
    data_yaml = "plan/data.yaml"
    separator = ","
    if isfile(data_yaml):
        with open(data_yaml, "r", encoding="utf-8") as f:
            for line in f:
                if separator in line:
                    key, val = line.split(separator, maxsplit=1)
                    d[key] = val.strip()

    d[collaborator_name] = dir_path

    # Write the data.yaml
    if isfile(data_yaml):
        with open(data_yaml, "w", encoding="utf-8") as f:
            for key, val in d.items():
                f.write(f"{key}{separator}{val}\n")


@collaborator.command(name="calchash")
@option(
    "-j",
    "--data_path",
    type=ClickPath(exists=True),
    help=(
        "Path to directory containing sources.json file defining the data sources of the dataset. "
        "This file should contain a JSON object with the data sources to be registered. For 'local'"
        " type, 'params' must include: 'path'. For 's3' type, 'params' must include: 'uri', "
        "'access_key_env_name', 'secret_key_env_name', 'secret_name', and optionally 'endpoint'."
    ),
)
def calchash(data_path):
    """Generates a hash for the dataset whose data sources are defined in `datasources.json`.

    The hash, saved as `hash.txt` in the same directory, can later
    be used in custom data loaders to validate and track dataset consistency.
    """

    if data_path and is_directory_traversal(data_path):
        logger.error("Data path is out of the openfl workspace scope.")
        sys.exit(1)
    if not os.path.isdir(data_path):
        logger.error("The data path must be a directory.")
        sys.exit(1)

    datasources_json_path = os.path.join(data_path, "datasources.json")
    if not os.path.isfile(datasources_json_path):
        logger.error(
            "The directory must contain a file named 'datasources.json' at the first level."
        )
        sys.exit(1)
    with open(datasources_json_path, "r", encoding="utf-8") as file:
        data = file.read()
    vds = DataSourcesJsonParser.parse(data)
    root_hash = vds.create_dataset_hash()
    hash_file_path = os.path.join(data_path, "hash.txt")
    with open(hash_file_path, "w", encoding="utf-8") as hash_file:
        hash_file.write(root_hash)

    logger.info(f"Dataset hash calculated and saved to {hash_file_path}. Hash: {root_hash}")


@collaborator.command(name="generate-cert-request")
@option(
    "-n",
    "--collaborator_name",
    required=True,
    help="The certified common name of the collaborator.",
)
@option("-s", "--silent", help="Do not prompt", is_flag=True)
@option(
    "-x",
    "--skip-package",
    help="If set, does not package the certificate signing request for export.",
    is_flag=True,
)
def generate_cert_request_(collaborator_name, silent, skip_package):
    """
    Generates collaborator certificate key-pair
    to be sent to aggregator for signing.
    """
    generate_cert_request(collaborator_name, silent, skip_package)


def generate_cert_request(collaborator_name, silent, skip_package):
    """Create collaborator certificate key pair.

    Then create a package with the CSR to send for signing.

    Args:
        collaborator_name (str): The certified common name of the collaborator.
        silent (bool): Do not prompt.
        skip_package (bool): Do not package the certificate signing request
            for export.
    """

    common_name = f"{collaborator_name}".lower()
    subject_alternative_name = f"DNS:{common_name}"
    file_name = f"col_{common_name}"

    echo(
        f"Creating COLLABORATOR certificate key pair with following settings: "
        f"CN={style(common_name, fg='red')},"
        f" SAN={style(subject_alternative_name, fg='red')}"
    )

    client_private_key, client_csr = generate_csr(common_name, server=False)

    (CERT_DIR / "client").mkdir(parents=True, exist_ok=True)

    echo("  Moving COLLABORATOR certificate to: " + style(f"{CERT_DIR}/{file_name}", fg="green"))

    # Print csr hash before writing csr to disk
    csr_hash = get_csr_hash(client_csr)
    echo("The CSR Hash " + style(f"{csr_hash}", fg="red"))

    # Write collaborator csr and key to disk
    write_crt(client_csr, CERT_DIR / "client" / f"{file_name}.csr")
    write_key(client_private_key, CERT_DIR / "client" / f"{file_name}.key")

    if not skip_package:
        archive_type = "zip"
        archive_name = f"col_{common_name}_to_agg_cert_request"
        archive_file_name = archive_name + "." + archive_type

        # Collaborator certificate signing request
        tmp_dir = join(mkdtemp(), "openfl", archive_name)

        ignore = ignore_patterns("__pycache__", "*.key", "*.srl", "*.pem")
        # Copy the current directory into the temporary directory
        copytree(f"{CERT_DIR}/client", tmp_dir, ignore=ignore)

        for f in glob(f"{tmp_dir}/*"):
            if common_name not in basename(f):
                remove(f)

        # Create Zip archive of directory
        make_archive(archive_name, archive_type, tmp_dir)
        rmtree(tmp_dir)

        echo(f"Archive {archive_file_name} with certificate signing request created")
        echo(
            "This file should be sent to the certificate authority"
            " (typically hosted by the aggregator) for signing"
        )


def find_certificate_name(file_name):
    """Parse the collaborator name.

    Args:
        file_name (str): The name of the collaborator in this federation.

    Returns:
        col_name (str): The collaborator name.
    """
    col_name = str(file_name).split(os.sep)[-1].split(".")[0][4:]
    return col_name


def register_collaborator(file_name):
    """Register the collaborator name in the cols.yaml list.

    Args:
        file_name (str): The name of the collaborator in this federation.
    """

    col_name = find_certificate_name(file_name)

    cols_file = Path("plan/cols.yaml").absolute()

    if not isfile(cols_file):
        cols_file.touch()
    with open(cols_file, "r", encoding="utf-8") as f:
        doc = load(f, Loader=FullLoader)

    if not doc:  # YAML is not correctly formatted
        doc = {}  # Create empty dictionary

    # List doesn't exist
    if "collaborators" not in doc.keys() or not doc["collaborators"]:
        doc["collaborators"] = []  # Create empty list

    if col_name in doc["collaborators"]:
        echo(
            "\nCollaborator "
            + style(f"{col_name}", fg="green")
            + " is already in the "
            + style(f"{cols_file}", fg="green")
        )

    else:
        doc["collaborators"].append(col_name)
        with open(cols_file, "w", encoding="utf-8") as f:
            dump(doc, f)

        echo(
            "\nRegistering "
            + style(f"{col_name}", fg="green")
            + " in "
            + style(f"{cols_file}", fg="green")
        )


@collaborator.command(name="certify")
@option("-n", "--collaborator_name", help="The certified common name of the collaborator.")
@option("-s", "--silent", help="Do not prompt", is_flag=True)
@option(
    "-r",
    "--request-pkg",
    type=ClickPath(exists=True),
    help="Path to a zip containing the certificate signing request for a collaborator.",
)
@option(
    "-i",
    "--import",
    "import_",
    type=ClickPath(exists=True),
    help="Path to an archive containing the collaborator's certificate (signed by the CA).",
)
def certify_(collaborator_name, silent, request_pkg, import_):
    """Certifies the collaborator certificate key-pair."""
    certify(collaborator_name, silent, request_pkg, import_)


def certify(collaborator_name, silent, request_pkg=None, import_=False):
    """Sign/certify collaborator certificate key pair.

    Args:
        collaborator_name (str): The certified common name of the collaborator.
        silent (bool): Do not prompt.
        request_pkg (str, optional): The archive containing the certificate
            signing request (*.zip) for a collaborator. Defaults to None.
        import_ (bool, optional): Import the archive containing the
            collaborator's certificate (signed by the CA). Defaults to False.
    """

    common_name = f"{collaborator_name}".lower()

    if not import_:
        if request_pkg:
            Path(f"{CERT_DIR}/client").mkdir(parents=True, exist_ok=True)
            unpack_archive(request_pkg, extract_dir=f"{CERT_DIR}/client")
            csr = glob(f"{CERT_DIR}/client/*.csr")[0]
        else:
            if collaborator_name is None:
                echo(
                    "collaborator_name can only be omitted if signing\n"
                    "a zipped request package.\n"
                    "\n"
                    "Example: fx collaborator certify --request-pkg "
                    "col_one_to_agg_cert_request.zip"
                )
                return
            csr = glob(f"{CERT_DIR}/client/col_{common_name}.csr")[0]
            copy(csr, CERT_DIR)
        cert_name = splitext(csr)[0]
        file_name = basename(cert_name)
        signing_key_path = "ca/signing-ca/private/signing-ca.key"
        signing_crt_path = "ca/signing-ca.crt"

        # Load CSR
        if not Path(f"{cert_name}.csr").exists():
            echo(
                style(
                    "Collaborator certificate signing request not found.",
                    fg="red",
                )
                + " Please run `fx collaborator generate-cert-request`"
                " to generate the certificate request."
            )

        csr, csr_hash = read_csr(f"{cert_name}.csr")

        # Load private signing key
        if not Path(CERT_DIR / signing_key_path).exists():
            echo(
                style("Signing key not found.", fg="red") + " Please run `fx workspace certify`"
                " to initialize the local certificate authority."
            )

        signing_key = read_key(CERT_DIR / signing_key_path)

        # Load signing cert
        if not Path(CERT_DIR / signing_crt_path).exists():
            echo(
                style("Signing certificate not found.", fg="red")
                + " Please run `fx workspace certify`"
                " to initialize the local certificate authority."
            )

        signing_crt = read_crt(CERT_DIR / signing_crt_path)

        echo(f"The CSR Hash for file {file_name}.csr is {csr_hash}")

        if silent:
            echo(
                "Signing COLLABORATOR certificate, "
                "Warning: manual check of certificate hashes is bypassed in silent mode."
            )
            signed_col_cert = sign_certificate(csr, signing_key, signing_crt.subject)
            write_crt(signed_col_cert, f"{cert_name}.crt")
            register_collaborator(CERT_DIR / "client" / f"{file_name}.crt")

        else:
            echo("Make sure the two hashes above are the same.")
            if confirm("Do you want to sign this certificate?"):
                echo(" Signing COLLABORATOR certificate")
                signed_col_cert = sign_certificate(csr, signing_key, signing_crt.subject)
                write_crt(signed_col_cert, f"{cert_name}.crt")
                register_collaborator(CERT_DIR / "client" / f"{file_name}.crt")

            else:
                echo(
                    style("Not signing certificate.", fg="red")
                    + " Please check with this collaborator to get the"
                    " correct certificate for this federation."
                )
                return

        if len(common_name) == 0:
            # If the collaborator name is provided, the collaborator and
            # certificate does not need to be exported
            return

        # Remove unneeded CSR
        remove(f"{cert_name}.csr")

        archive_type = "zip"
        archive_name = f"agg_to_{file_name}_signed_cert"

        # Collaborator certificate signing request
        tmp_dir = join(mkdtemp(), "openfl", archive_name)

        Path(f"{tmp_dir}/client").mkdir(parents=True, exist_ok=True)
        # Copy the signed cert to the temporary directory
        copy(f"{CERT_DIR}/client/{file_name}.crt", f"{tmp_dir}/client/")
        # Copy the CA certificate chain to the temporary directory
        copy(f"{CERT_DIR}/cert_chain.crt", tmp_dir)

        # Create Zip archive of directory
        make_archive(archive_name, archive_type, tmp_dir)
        rmtree(tmp_dir)

    else:
        _import_certificates(import_)


def _import_certificates(archive: str):
    # Copy the signed certificate and cert chain into PKI_DIR
    previous_crts = glob(f"{CERT_DIR}/client/*.crt")
    unpack_archive(archive, extract_dir=CERT_DIR)
    updated_crts = glob(f"{CERT_DIR}/client/*.crt")
    cert_difference = list(set(updated_crts) - set(previous_crts))
    if len(cert_difference) != 0:
        crt = basename(cert_difference[0])
        echo(f"Certificate {crt} installed to PKI directory")
    else:
        echo("Certificate updated in the PKI directory")
