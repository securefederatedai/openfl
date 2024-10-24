# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Workspace module."""
import logging
import os
import shutil
import subprocess  # nosec
import sys
import tempfile
from hashlib import sha256
from os import chdir
from os.path import basename, isfile
from pathlib import Path
from shutil import copyfile, copytree, ignore_patterns, unpack_archive
from subprocess import check_call  # nosec
from sys import executable
from typing import Tuple, Union

from click import Choice
from click import Path as ClickPath
from click import echo, group, option, pass_context
from cryptography.hazmat.primitives import serialization

from openfl.cryptography.ca import generate_root_cert, generate_signing_csr, sign_certificate
from openfl.federated.plan import Plan
from openfl.interface import plan
from openfl.interface.cli_helper import CERT_DIR, OPENFL_USERDIR, SITEPACKS, WORKSPACE, print_tree


@group()
@pass_context
def workspace(context):
    """Manage Federated Learning Workspaces.

    Args:
        context: The context in which the command is being invoked.
    """
    context.obj["group"] = "workspace"


def is_directory_traversal(directory: Union[str, Path]) -> bool:
    """Check for directory traversal.

    Args:
        directory (Union[str, Path]): The directory to check.

    Returns:
        bool: True if directory traversal is detected, False otherwise.
    """
    cwd = os.path.abspath(os.getcwd())
    requested_path = os.path.relpath(directory, start=cwd)
    requested_path = os.path.abspath(requested_path)
    common_prefix = os.path.commonprefix([requested_path, cwd])
    return common_prefix != cwd


def create_dirs(prefix):
    """Create workspace directories.

    Args:
        prefix: The prefix for the directories to be created.
    """

    echo("Creating Workspace Directories")

    (prefix / "cert").mkdir(parents=True, exist_ok=True)  # certifications
    (prefix / "data").mkdir(parents=True, exist_ok=True)  # training data
    (prefix / "logs").mkdir(parents=True, exist_ok=True)  # training logs
    (prefix / "save").mkdir(parents=True, exist_ok=True)  # model weight saves / initialization
    (prefix / "src").mkdir(parents=True, exist_ok=True)  # model code

    copyfile(WORKSPACE / "workspace" / ".workspace", prefix / ".workspace")


def create_temp(prefix, template):
    """Create workspace templates.

    Args:
        prefix: The prefix for the directories to be created.
        template: The template to use for creating the workspace.
    """

    echo("Creating Workspace Templates")

    copytree(
        src=WORKSPACE / template,
        dst=prefix,
        dirs_exist_ok=True,
        ignore=ignore_patterns("__pycache__"),
    )  # from template workspace


def get_templates():
    """Grab the default templates from the distribution.

    Returns:
        list: A list of default templates.
    """

    return [
        d.name
        for d in WORKSPACE.glob("*")
        if d.is_dir() and d.name not in ["__pycache__", "workspace", "experimental"]
    ]


@workspace.command(name="create")
@option("--prefix", required=True, help="Workspace name or path", type=ClickPath())
@option("--template", required=True, type=Choice(get_templates()))
def create_(prefix, template):
    """Create the workspace.

    Args:
        prefix: The prefix for the directories to be created.
        template: The template to use for creating the workspace.
    """
    if is_directory_traversal(prefix):
        echo("Workspace name or path is out of the openfl workspace scope.")
        sys.exit(1)
    create(prefix, template)


def create(prefix, template):
    """Create federated learning workspace.

    Args:
        prefix: The prefix for the directories to be created.
        template: The template to use for creating the workspace.
    """

    if not OPENFL_USERDIR.exists():
        OPENFL_USERDIR.mkdir()

    prefix = Path(prefix).absolute()

    create_dirs(prefix)
    create_temp(prefix, template)

    requirements_filename = "requirements.txt"

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

    apply_template_plan(prefix, template)

    print_tree(prefix, level=3)


@workspace.command(name="import")
@option(
    "--archive",
    required=True,
    help="Zip file containing workspace to import",
    type=ClickPath(exists=True),
)
def import_(archive):
    """Import federated learning workspace.

    Args:
        archive: The archive file containing the workspace to import.
    """

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
            [executable, "-m", "pip", "install", "-r", "requirements.txt"],
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


def _get_dir_hash(path):
    """Get the hash of a directory.

    Args:
        path (str): The path of the directory.

    Returns:
        str: The hash of the directory.
    """
    hash_ = sha256()
    hash_.update(path.encode("utf-8"))
    hash_ = hash_.hexdigest()
    return hash_


# Commands for workspace packaging and distribution
# -------------------------------------------------


@workspace.command(name="export")
def export_() -> str:
    """
    Exports the OpenFL workspace (in current directory)
    to an archive.

    \b
    The archive contains the following files/dirs copied as-is:
        - `src`: All experiment source code.
        - `plan`: The FL plan directory.
        - `save`: Model initial weights.
        - `requirements.txt`: Package list required for the experiment.

    This archive does *not* copy `data`, `logs`, or secrets.

    This command takes no arguments.
    """
    plan_file = os.path.abspath(os.path.join("plan", "plan.yaml"))
    if not os.path.isfile(plan_file):
        raise FileNotFoundError(
            f"{plan_file} does not exist in the current directory.\n"
            "Please ensure this command is being run from a workspace."
        )
    plan.freeze_plan(plan_file)

    # Create a staging area.
    workspace_name = os.path.basename(os.getcwd())
    tmp_dir = os.path.join(tempfile.mkdtemp(), "openfl", workspace_name)
    ignore = shutil.ignore_patterns(
        *["__pycache__", "*.crt", "*.key", "*.csr", "*.srl", "*.pem", "*.pbuf"]
    )

    # Export the minimum required files to set up a collaborator
    # os.makedirs(os.path.join(tmp_dir, 'save'), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "data"), exist_ok=True)
    shutil.copytree("src", os.path.join(tmp_dir, "src"), ignore=ignore)
    shutil.copytree("plan", os.path.join(tmp_dir, "plan"), ignore=ignore)
    shutil.copytree("save", os.path.join(tmp_dir, "save"))
    shutil.copy2("requirements.txt", os.path.join(tmp_dir, "requirements.txt"))

    _ws_identifier_file = ".workspace"
    if not os.path.isfile(_ws_identifier_file):
        openfl_ws_identifier_file = os.path.join(WORKSPACE, "workspace", _ws_identifier_file)
        logging.warning(
            f"`{_ws_identifier_file}` is missing, " f"copying {openfl_ws_identifier_file} as-is."
        )
        shutil.copy2(openfl_ws_identifier_file, tmp_dir)
    shutil.copy2(_ws_identifier_file, tmp_dir)

    # Create Zip archive of directory
    _ARCHIVE_FORMAT = "zip"
    shutil.make_archive(workspace_name, _ARCHIVE_FORMAT, tmp_dir)
    archive = f"{workspace_name}.{_ARCHIVE_FORMAT}"
    logging.info(f"Export: {archive} created")
    return archive


@workspace.command(name="dockerize")
@option(
    "--save",
    is_flag=True,
    default=False,
    help="Export the docker image as <workspace_name>.tar file.",
)
@option(
    "--rebuild",
    is_flag=True,
    default=False,
    help="If set, rebuilds docker images with `--no-cache` option.",
)
@option(
    "--revision",
    required=False,
    default=None,
    help=(
        "Optional, version of OpenFL source code to build base image from. "
        "If unspecified, default value in `Dockerfile.base` will be used, "
        "typically the latest stable release. "
        "Format: <OPENFL_GIT_URL>@<COMMIT_ID/BRANCH>"
    ),
)
@pass_context
def dockerize_(context, save, rebuild, revision):
    """Package current workspace as a Docker image."""

    # Docker build options
    options = []
    options.append("--no-cache" if rebuild else "")
    options.append(f"--build-arg OPENFL_REVISION={revision}" if revision else "")
    options = " ".join(options)

    # Export workspace
    archive = context.invoke(export_)
    workspace_name, _ = archive.split(".")

    # Build OpenFL base image.
    logging.info("Building OpenFL Base image")
    base_image_build_cmd = (
        "DOCKER_BUILDKIT=1 docker build {options} "
        "-t {image_name} "
        "-f {dockerfile} "
        "{build_context}"
    ).format(
        options=options,
        image_name="openfl",
        dockerfile=os.path.join(SITEPACKS, "openfl-docker", "Dockerfile.base"),
        build_context=".",
    )
    _execute(base_image_build_cmd)

    # Build workspace image.
    logging.info("Building workspace image")
    ws_image_build_cmd = (
        "DOCKER_BUILDKIT=1 docker build {options} "
        "--build-arg WORKSPACE_NAME={workspace_name} "
        "-t {image_name} "
        "-f {dockerfile} "
        "{build_context}"
    ).format(
        options=options,
        image_name=workspace_name,
        workspace_name=workspace_name,
        dockerfile=os.path.join(SITEPACKS, "openfl-docker", "Dockerfile.workspace"),
        build_context=".",
    )
    _execute(ws_image_build_cmd)

    # Export workspace as tarball (optional)
    if save:
        logging.info("Saving workspace docker image...")
        save_image_cmd = "docker save {image_name} -o {image_name}.tar"
        _execute(save_image_cmd.format(image_name=workspace_name))
        logging.info(f"Docker image saved to file: {workspace_name}.tar")


@workspace.command(name="graminize")
@option(
    "-s",
    "--signing-key",
    required=False,
    type=lambda p: Path(p).absolute(),
    default="/",
    help="A 3072-bit RSA private key (PEM format) is required for signing the manifest.\n"
    "If a key is passed the gramine-sgx manifest fill be prepared.\n"
    "In option is ignored this command will build an image that can only run "
    "with gramine-direct (not in enclave).",
)
@option(
    "-e",
    "--enclave_size",
    required=False,
    type=str,
    default="16G",
    help="Memory size of the enclave, defined as number with size suffix. "
    "Must be a power-of-2.\n"
    "Default is 16G.",
)
@option(
    "-t",
    "--tag",
    required=False,
    type=str,
    multiple=False,
    default="",
    help="Tag of the built image.\n" "By default, the workspace name is used.",
)
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
@option(
    "--save/--no-save",
    required=False,
    default=True,
    type=bool,
    help="Dump the Docker image to an archive",
)
@option("--rebuild", help="Build images with `--no-cache`", is_flag=True)
@pass_context
def graminize_(
    context,
    signing_key: Path,
    enclave_size: str,
    tag: str,
    pip_install_options: Tuple[str],
    save: bool,
    rebuild: bool,
) -> None:
    """
    Build gramine app inside a docker image.

    This command is the alternative to `workspace export`.
    It should be called after `fx plan initialize` inside the workspace dir.

    User is expected to be in docker group.
    If your machine is behind a proxy, make sure you set it up in
    ~/.docker/config.json.

    TODO:
    1. gramine-direct, check if a key is provided
    2. make a standalone function with `export` parametr

    Args:
        context: The context in which the command is being invoked.
        signing_key (Path): A 3072-bit RSA private key (PEM format) is
            required for signing the manifest.
        enclave_size (str): Memory size of the enclave, defined as number with
            size suffix.
        tag (str): Tag of the built image.
        pip_install_options (Tuple[str]): Options for remote pip install.
        save (bool): Whether to dump the Docker image to an archive.
        rebuild (bool): Whether to build images with `--no-cache`.
    """

    def open_pipe(command: str):
        echo(f"\n 📦 Executing command:\n{command}\n")
        process = subprocess.Popen(
            command,
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        )
        for line in process.stdout:
            echo(line)
        _ = process.communicate()  # pipe is already empty, used to get `returncode`
        if process.returncode != 0:
            raise Exception("\n ❌ Execution failed\n")

    # We can build for gramine-sgx and run with gramine-direct,
    # but not vice versa.
    sgx_build = signing_key.is_file()
    if sgx_build:
        echo("\n Building SGX-ready applecation")
    else:
        echo("\n Building gramine-direct applecation")
    rebuild_option = "--no-cache" if rebuild else ""

    os.environ["DOCKER_BUILDKIT"] = "1"

    echo("\n 🐋 Building base gramine-openfl image...")
    base_dockerfile = SITEPACKS / "openfl-gramine" / "Dockerfile.gramine"
    base_build_command = f"docker build {rebuild_option} -t gramine_openfl -f {base_dockerfile} ."
    open_pipe(base_build_command)
    echo("\n ✔️ DONE: Building base gramine-openfl image")

    workspace_path = Path.cwd()
    workspace_name = workspace_path.name

    if not tag:
        tag = workspace_name

    context.invoke(export_, pip_install_options=pip_install_options)
    workspace_archive = workspace_path / f"{workspace_name}.zip"

    grainized_ws_dockerfile = SITEPACKS / "openfl-gramine" / "Dockerfile.graminized.workspace"

    echo("\n 🐋 Building graminized workspace image...")
    signing_key = f"--secret id=signer-key,src={signing_key} " if sgx_build else ""
    graminized_build_command = (
        f"docker build -t {tag} {rebuild_option} "
        "--build-arg BASE_IMAGE=gramine_openfl "
        f"--build-arg WORKSPACE_ARCHIVE={workspace_archive.relative_to(workspace_path)} "
        f"--build-arg SGX_ENCLAVE_SIZE={enclave_size} "
        f"--build-arg SGX_BUILD={int(sgx_build)} "
        f"{signing_key}"
        f"-f {grainized_ws_dockerfile} {workspace_path}"
    )
    open_pipe(graminized_build_command)
    echo("\n ✔️ DONE: Building graminized workspace image")

    if save:
        echo("\n 💾 Saving the graminized workspace image...")
        save_image_command = f"docker save {tag} | gzip > {tag}.tar.gz"
        open_pipe(save_image_command)
        echo(f"\n ✔️ The image saved to file: {tag}.tar.gz")


def apply_template_plan(prefix, template):
    """Copy plan file from template folder.

    This function unfolds default values from template plan configuration
    and writes the configuration to the current workspace.

    Args:
        prefix: The prefix for the directories to be created.
        template: The template to use for creating the workspace.
    """

    template_plan = Plan.parse(WORKSPACE / template / "plan" / "plan.yaml")

    Plan.dump(prefix / "plan" / "plan.yaml", template_plan.config)


def _execute(cmd: str, verbose=True) -> None:
    """Executes `cmd` as a subprocess

    Args:
        cmd (str): Command to be executed.

    Raises:
        Exception: If return code is nonzero

    Returns:
        `stdout` of the command as list of messages
    """
    logging.info(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    stdout_log = list()
    for line in process.stdout:
        msg = line.rstrip().decode("utf-8")
        stdout_log.append(msg)
        if verbose:
            logging.info(msg)

    process.communicate()
    if process.returncode != 0:
        raise Exception(f"`{cmd}` failed with return code {process.returncode}")

    return stdout_log
