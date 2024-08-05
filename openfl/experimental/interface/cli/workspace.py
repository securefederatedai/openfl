# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Workspace module."""
import os
import sys
from hashlib import sha256
from logging import getLogger
from os import chdir, getcwd, makedirs
from os.path import basename, isfile, join
from pathlib import Path
from shutil import copy2, copyfile, copytree, ignore_patterns, make_archive, unpack_archive
from subprocess import check_call
from sys import executable
from tempfile import mkdtemp
from typing import Tuple

from click import Choice
from click import Path as ClickPath
from click import confirm, echo, group, option, pass_context, style
from cryptography.hazmat.primitives import serialization

from openfl.cryptography.ca import generate_root_cert, generate_signing_csr, sign_certificate
from openfl.experimental.federated.plan import Plan
from openfl.experimental.interface.cli.cli_helper import (
    CERT_DIR,
    OPENFL_USERDIR,
    WORKSPACE,
    print_tree,
)
from openfl.experimental.interface.cli.plan import freeze_plan
from openfl.experimental.workspace_export import WorkspaceExport
from openfl.utilities.path_check import is_directory_traversal
from openfl.utilities.utils import rmtree
from openfl.utilities.workspace import dump_requirements_file

logger = getLogger(__name__)


@group()
@pass_context
def workspace(context):
    """Manage Experimental Federated Learning Workspaces."""
    context.obj["group"] = "workspace"


def create_dirs(prefix):
    """Create workspace directories."""

    echo("Creating Workspace Directories")

    (prefix / "cert").mkdir(parents=True, exist_ok=True)  # certifications
    (prefix / "data").mkdir(parents=True, exist_ok=True)  # training data
    (prefix / "logs").mkdir(parents=True, exist_ok=True)  # training logs
    (prefix / "save").mkdir(parents=True, exist_ok=True)  # model weight saves / initialization
    (prefix / "src").mkdir(parents=True, exist_ok=True)  # model code

    copyfile(WORKSPACE / "workspace" / ".workspace", prefix / ".workspace")


def create_temp(prefix, template):
    """Create workspace templates."""

    echo("Creating Workspace Templates")
    # Use the specified template if it's a Path, otherwise use WORKSPACE/template
    source = template if isinstance(template, Path) else WORKSPACE / template

    copytree(
        src=source,
        dst=prefix,
        dirs_exist_ok=True,
        ignore=ignore_patterns("__pycache__"),
    )  # from template workspace
    apply_template_plan(prefix, template)


def get_templates():
    """Grab the default templates from the distribution."""

    return [
        d.name
        for d in WORKSPACE.glob("*")
        if d.is_dir() and d.name not in ["__pycache__", "workspace"]
    ]


@workspace.command(name="create")
@option("--prefix", required=True, help="Workspace name or path", type=ClickPath())
@option(
    "--custom_template",
    required=False,
    help="Path to custom template",
    type=ClickPath(exists=True),
)
@option(
    "--notebook",
    required=False,
    help="Path to jupyter notebook",
    type=ClickPath(exists=True),
)
@option(
    "--template_output_dir",
    required=False,
    help="Destination directory to save your Jupyter Notebook workspace.",
    type=ClickPath(exists=False, file_okay=False, dir_okay=True),
)
@option("--template", required=False, type=Choice(get_templates()))
def create_(prefix, custom_template, template, notebook, template_output_dir):
    """Create the experimental workspace."""
    if is_directory_traversal(prefix):
        echo("Workspace name or path is out of the openfl workspace scope.")
        sys.exit(1)

    if custom_template and template and notebook:
        raise ValueError(
            "Please provide either `template`, `custom_template` or "
            + "`notebook`. Not all are necessary"
        )
    elif (
        (custom_template and template) or (template and notebook) or (custom_template and notebook)
    ):
        raise ValueError(
            "Please provide only one of the following options: "
            + "`template`, `custom_template`, or `notebook`."
        )

    if not (custom_template or template or notebook):
        raise ValueError(
            "Please provide one of the following options: "
            + "`template`, `custom_template`, or `notebook`."
        )

    if notebook:
        if not template_output_dir:
            raise ValueError(
                "Please provide output_workspace which is Destination directory to "
                + "save your Jupyter Notebook workspace."
            )

        WorkspaceExport.export(
            notebook_path=notebook,
            output_workspace=template_output_dir,
        )

        create(prefix, Path(template_output_dir).resolve())

        logger.warning(
            "The user should review the generated workspace for completeness " + "before proceeding"
        )
    else:
        template = Path(custom_template).resolve() if custom_template else template
        create(prefix, template)


def create(prefix, template):
    """Create federated learning workspace."""

    if not OPENFL_USERDIR.exists():
        OPENFL_USERDIR.mkdir()

    prefix = Path(prefix).absolute()

    create_dirs(prefix)
    create_temp(prefix, template)

    requirements_filename = "requirements.txt"

    if not os.path.exists(f"{str(prefix)}/plan/data.yaml"):
        echo(
            style(
                "Participant private attributes shall be set to None as plan/data.yaml"
                + " was not found in the workspace.",
                fg="yellow",
            )
        )

    if isfile(f"{str(prefix)}/{requirements_filename}"):
        check_call(
            [
                executable,
                "-m",
                "pip",
                "install",
                "-r",
                f"{prefix}/requirements.txt",
            ],
            shell=False,
        )
        echo(f"Successfully installed packages from {prefix}/requirements.txt.")
    else:
        echo("No additional requirements for workspace defined. Skipping...")
    prefix_hash = _get_dir_hash(str(prefix.absolute()))
    with open(
        OPENFL_USERDIR / f"requirements.{prefix_hash}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        check_call([executable, "-m", "pip", "freeze"], shell=False, stdout=f)

    print_tree(prefix, level=3)


@workspace.command(name="export")
@option(
    "-o",
    "--pip-install-options",
    required=False,
    type=str,
    multiple=True,
    default=tuple,
    help="Options for remote pip install. "
    "You may pass several options in quotation marks alongside with arguments, "
    'e.g. -o "--find-links source.site"',
)
def export_(pip_install_options: Tuple[str]):
    """Export federated learning workspace."""

    echo(
        style(
            "This command will archive the contents of 'plan' and 'src' directory, user"
            + " should review that these does not contain any information which is private and"
            + " not to be shared.",
            fg="yellow",
        )
    )

    plan_file = Path("plan/plan.yaml").absolute()
    try:
        freeze_plan(plan_file)
    except FileNotFoundError:
        echo(f'Plan file "{plan_file}" not found. No freeze performed.')

    # Dump requirements.txt
    dump_requirements_file(prefixes=pip_install_options, keep_original_prefixes=True)

    archive_type = "zip"
    archive_name = basename(getcwd())
    archive_file_name = archive_name + "." + archive_type

    # Aggregator workspace
    tmp_dir = join(mkdtemp(), "openfl", archive_name)

    ignore = ignore_patterns("__pycache__", "*.crt", "*.key", "*.csr", "*.srl", "*.pem", "*.pbuf")

    # We only export the minimum required files to set up a collaborator
    makedirs(f"{tmp_dir}/save", exist_ok=True)
    makedirs(f"{tmp_dir}/logs", exist_ok=True)
    makedirs(f"{tmp_dir}/data", exist_ok=True)
    copytree("./src", f"{tmp_dir}/src", ignore=ignore)  # code
    copytree("./plan", f"{tmp_dir}/plan", ignore=ignore)  # plan
    copy2("./requirements.txt", f"{tmp_dir}/requirements.txt")  # requirements

    try:
        copy2(".workspace", tmp_dir)  # .workspace
    except FileNotFoundError:
        echo("'.workspace' file not found.")
        if confirm("Create a default '.workspace' file?"):
            copy2(WORKSPACE / "workspace" / ".workspace", tmp_dir)
        else:
            echo("To proceed, you must have a '.workspace' " "file in the current directory.")
            raise

    # Create Zip archive of directory
    echo("\n 🗜️ Preparing workspace distribution zip file")
    make_archive(archive_name, archive_type, tmp_dir)
    rmtree(tmp_dir)
    echo(f"\n ✔️ Workspace exported to archive: {archive_file_name}")


@workspace.command(name="import")
@option(
    "--archive",
    required=True,
    help="Zip file containing workspace to import",
    type=ClickPath(exists=True),
)
def import_(archive):
    """Import federated learning workspace."""

    archive = Path(archive).absolute()

    dir_path = basename(archive).split(".")[0]
    unpack_archive(archive, extract_dir=dir_path)
    chdir(dir_path)

    requirements_filename = "requirements.txt"

    if isfile(requirements_filename):
        check_call(
            [executable, "-m", "pip", "install", "--upgrade", "pip"],
            shell=False,
        )
        check_call(
            [executable, "-m", "pip", "install", "-r", requirements_filename],
            shell=False,
        )
    else:
        echo("No " + requirements_filename + " file found.")

    echo(f"Workspace {archive} has been imported.")
    echo("You may need to copy your PKI certificates to join the federation.")


@workspace.command(name="certify")
def certify_():
    """Create certificate authority for federation."""
    certify()


def certify():
    """Create certificate authority for federation."""

    echo("Setting Up Certificate Authority...\n")

    echo("1.  Create Root CA")
    echo("1.1 Create Directories")

    (CERT_DIR / "ca/root-ca/private").mkdir(parents=True, exist_ok=True, mode=0o700)
    (CERT_DIR / "ca/root-ca/db").mkdir(parents=True, exist_ok=True)

    echo("1.2 Create Database")

    with open(CERT_DIR / "ca/root-ca/db/root-ca.db", "w", encoding="utf-8") as f:
        pass  # write empty file
    with open(CERT_DIR / "ca/root-ca/db/root-ca.db.attr", "w", encoding="utf-8") as f:
        pass  # write empty file

    with open(CERT_DIR / "ca/root-ca/db/root-ca.crt.srl", "w", encoding="utf-8") as f:
        f.write("01")  # write file with '01'
    with open(CERT_DIR / "ca/root-ca/db/root-ca.crl.srl", "w", encoding="utf-8") as f:
        f.write("01")  # write file with '01'

    echo("1.3 Create CA Request and Certificate")

    root_crt_path = "ca/root-ca.crt"
    root_key_path = "ca/root-ca/private/root-ca.key"

    root_private_key, root_cert = generate_root_cert()

    # Write root CA certificate to disk
    with open(CERT_DIR / root_crt_path, "wb") as f:
        f.write(
            root_cert.public_bytes(
                encoding=serialization.Encoding.PEM,
            )
        )

    with open(CERT_DIR / root_key_path, "wb") as f:
        f.write(
            root_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    echo("2.  Create Signing Certificate")
    echo("2.1 Create Directories")

    (CERT_DIR / "ca/signing-ca/private").mkdir(parents=True, exist_ok=True, mode=0o700)
    (CERT_DIR / "ca/signing-ca/db").mkdir(parents=True, exist_ok=True)

    echo("2.2 Create Database")

    with open(CERT_DIR / "ca/signing-ca/db/signing-ca.db", "w", encoding="utf-8") as f:
        pass  # write empty file
    with open(CERT_DIR / "ca/signing-ca/db/signing-ca.db.attr", "w", encoding="utf-8") as f:
        pass  # write empty file

    with open(CERT_DIR / "ca/signing-ca/db/signing-ca.crt.srl", "w", encoding="utf-8") as f:
        f.write("01")  # write file with '01'
    with open(CERT_DIR / "ca/signing-ca/db/signing-ca.crl.srl", "w", encoding="utf-8") as f:
        f.write("01")  # write file with '01'

    echo("2.3 Create Signing Certificate CSR")

    signing_csr_path = "ca/signing-ca.csr"
    signing_crt_path = "ca/signing-ca.crt"
    signing_key_path = "ca/signing-ca/private/signing-ca.key"

    signing_private_key, signing_csr = generate_signing_csr()

    # Write Signing CA CSR to disk
    with open(CERT_DIR / signing_csr_path, "wb") as f:
        f.write(
            signing_csr.public_bytes(
                encoding=serialization.Encoding.PEM,
            )
        )

    with open(CERT_DIR / signing_key_path, "wb") as f:
        f.write(
            signing_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    echo("2.4 Sign Signing Certificate CSR")

    signing_cert = sign_certificate(signing_csr, root_private_key, root_cert.subject, ca=True)

    with open(CERT_DIR / signing_crt_path, "wb") as f:
        f.write(
            signing_cert.public_bytes(
                encoding=serialization.Encoding.PEM,
            )
        )

    echo("3   Create Certificate Chain")

    # create certificate chain file by combining root-ca and signing-ca
    with open(CERT_DIR / "cert_chain.crt", "w", encoding="utf-8") as d:
        with open(CERT_DIR / "ca/root-ca.crt", encoding="utf-8") as s:
            d.write(s.read())
        with open(CERT_DIR / "ca/signing-ca.crt") as s:
            d.write(s.read())

    echo("\nDone.")


# FIXME: Function is not in use


def _get_requirements_dict(txtfile):
    with open(txtfile, "r", encoding="utf-8") as snapshot:
        snapshot_dict = {}
        for line in snapshot:
            try:
                # 'pip freeze' generates requirements with exact versions
                k, v = line.split("==")
                snapshot_dict[k] = v
            except ValueError:
                snapshot_dict[line] = None
        return snapshot_dict


def _get_dir_hash(path):

    hash_ = sha256()
    hash_.update(path.encode("utf-8"))
    hash_ = hash_.hexdigest()
    return hash_


def apply_template_plan(prefix, template):
    """Copy plan file from template folder.

    This function unfolds default values from template plan configuration and
    writes the configuration to the current workspace.
    """

    # Use the specified template if it's a Path, otherwise use
    # WORKSPACE/template
    source = template if isinstance(template, Path) else WORKSPACE / template

    template_plan = Plan.parse(source / "plan" / "plan.yaml")

    Plan.dump(prefix / "plan" / "plan.yaml", template_plan.config)
