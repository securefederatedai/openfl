# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Workspace module."""

from pathlib import Path
from click import Choice, Path as ClickPath
from click import group, option, pass_context
from click import echo, confirm
from subprocess import check_call
from sys import executable
from warnings import warn
from shutil import copyfile, ignore_patterns

from openfl.interface.cli_helper import copytree, print_tree, check_varenv
from openfl.interface.cli_helper import get_fx_path, replace_line_in_file
from openfl.interface.cli_helper import WORKSPACE, PKI_DIR
from openfl.interface.cli_helper import SITEPACKS, OPENFL_USERDIR


@group()
@pass_context
def workspace(context):
    """Manage Federated Learning Workspaces."""
    context.obj['group'] = 'workspace'


def create_dirs(prefix):
    """Create workspace directories."""
    echo('Creating Workspace Directories')

    (prefix / 'cert').mkdir(parents=True, exist_ok=True)  # certifications
    (prefix / 'data').mkdir(parents=True, exist_ok=True)  # training data
    (prefix / 'logs').mkdir(parents=True, exist_ok=True)  # training logs
    (prefix / 'plan').mkdir(parents=True, exist_ok=True)  # federated learning plans
    (prefix / 'save').mkdir(parents=True, exist_ok=True)  # model weight saves / initialization
    (prefix / 'code').mkdir(parents=True, exist_ok=True)  # model code

    src = WORKSPACE / 'workspace/plan/defaults'  # from default workspace
    dst = prefix / 'plan/defaults'  # to created workspace

    copytree(src=src, dst=dst, dirs_exist_ok=True)
    copyfile(WORKSPACE / 'workspace' / '.workspace', prefix / '.workspace')


def create_temp(prefix, template):
    """Create workspace templates."""
    echo('Creating Workspace Templates')

    copytree(src=WORKSPACE / template, dst=prefix, dirs_exist_ok=True,
             ignore=ignore_patterns('__pycache__'))  # from template workspace


def get_templates():
    """Grab the default templates from the distribution."""
    return [d.name for d in WORKSPACE.glob('*') if d.is_dir()
            and d.name not in ['__pycache__', 'workspace']]


@workspace.command(name='create')
@option('--prefix', required=True,
        help='Workspace name or path', type=ClickPath())
@option('--template', required=True, type=Choice(get_templates()))
def create_(prefix, template):
    """Create the workspace."""
    create(prefix, template)


def create(prefix, template):
    """Create federated learning workspace."""
    from os.path import isfile
    if not OPENFL_USERDIR.exists():
        OPENFL_USERDIR.mkdir()

    prefix = Path(prefix)
    template = Path(template)

    create_dirs(prefix)
    create_temp(prefix, template)

    requirements_filename = "requirements.txt"

    if isfile(f'{str(prefix)}/{requirements_filename}'):
        check_call([
            executable, "-m", "pip", "install", "-r",
            f"{prefix}/requirements.txt"], shell=False)
        echo(f"Successfully installed packages from {prefix}/requirements.txt.")
    else:
        echo("No additional requirements for workspace defined. Skipping...")
    prefix_hash = _get_dir_hash(str(prefix.absolute()))
    with open(OPENFL_USERDIR / f'requirements.{prefix_hash}.txt', 'w') as f:
        check_call([executable, '-m', 'pip', 'freeze'], shell=False, stdout=f)

    print_tree(prefix, level=3)


@workspace.command(name='export')
@pass_context
def export_(context):
    """Export federated learning workspace."""
    from shutil import make_archive, copytree, copy2, ignore_patterns
    from tempfile import mkdtemp
    from os import getcwd, makedirs
    from os.path import basename, join
    from plan import FreezePlan

    # TODO: Does this need to freeze all plans?
    planFile = 'plan/plan.yaml'
    try:
        FreezePlan(planFile)
    except Exception:
        echo(f'Plan file "{planFile}" not found. No freeze performed.')

    requirements_filename = 'requirements.txt'
    workspace_reqs = _get_requirements_dict(requirements_filename)
    prefix = getcwd()

    export_requirements_filename = 'requirements.export.txt'
    with open(export_requirements_filename, "w") as f:
        check_call([executable, "-m", "pip", "freeze"], shell=False, stdout=f)
    workspace_hash = _get_dir_hash(prefix)
    origin_dict = _get_requirements_dict(
        OPENFL_USERDIR / f'requirements.{workspace_hash}.txt')
    current_dict = _get_requirements_dict(export_requirements_filename)
    with open(export_requirements_filename, "w") as f:
        for package, version in current_dict.items():
            if (
                package in workspace_reqs
                or package not in origin_dict
                or version != origin_dict[package]
            ):
                # we save only the difference between original workspace after
                # 'fx create workspace' and current one.
                echo(f'Writing {package}=={version} '
                     f'to {requirements_filename}...')
                f.write(f'{package}=={version}\n')
            elif version is None:  # local dependency
                warn(f'Could not generate requirements for {package}.'
                     f' Consider installing it manually after workspace'
                     f' import.')
    echo(f'{export_requirements_filename} written.')

    archiveType = 'zip'
    archiveName = basename(getcwd())
    archiveFileName = archiveName + '.' + archiveType

    # Aggregator workspace
    tmpDir = join(mkdtemp(), 'openfl', archiveName)

    ignore = ignore_patterns(
        '__pycache__', '*.crt', '*.key', '*.csr', '*.srl', '*.pem', '*.pbuf')

    # We only export the minimum required files to set up a collaborator
    makedirs(f'{tmpDir}/save', exist_ok=True)
    makedirs(f'{tmpDir}/logs', exist_ok=True)
    makedirs(f'{tmpDir}/data', exist_ok=True)
    copytree('./code', f'{tmpDir}/code', ignore=ignore)  # code
    copytree('./plan', f'{tmpDir}/plan', ignore=ignore)  # plan
    copy2(export_requirements_filename, f'{tmpDir}/requirements.txt')  # requirements

    try:
        copy2('.workspace', tmpDir)  # .workspace
    except FileNotFoundError:
        echo('\'.workspace\' file not found.')
        if confirm('Create a default \'.workspace\' file?'):
            copy2(WORKSPACE / 'workspace' / '.workspace', tmpDir)
        else:
            echo('To proceed, you must have a \'.workspace\' '
                 'file in the current directory.')
            raise

    # Create Zip archive of directory
    make_archive(archiveName, archiveType, tmpDir)

    echo(f'Workspace exported to archive: {archiveFileName}')


@workspace.command(name='import')
@option('--archive', required=True,
        help='Zip file containing workspace to import',
        type=ClickPath(exists=True))
def import_(archive):
    """Import federated learning workspace."""
    from shutil import unpack_archive
    from os.path import isfile, basename
    from os import chdir

    dirPath = basename(archive).split('.')[0]
    unpack_archive(archive, extract_dir=dirPath)
    chdir(dirPath)

    requirements_filename = "requirements.txt"

    if isfile(requirements_filename):
        check_call([
            executable, "-m", "pip", "install", "-r", "requirements.txt"],
            shell=False)
    else:
        echo("No " + requirements_filename + " file found.")

    echo(f'Workspace {archive} has been imported.')
    echo('You may need to copy your PKI certificates to join the federation.')


@workspace.command(name='certify')
def certify_():
    """Create certificate authority for federation."""
    certify()


def certify():
    """Create certificate authority for federation."""
    from openfl.cryptography.ca import generate_root_cert, generate_signing_csr, sign_certificate
    from cryptography.hazmat.primitives import serialization

    echo('Setting Up Certificate Authority...\n')

    echo('1.  Create Root CA')
    echo('1.1 Create Directories')

    (PKI_DIR / 'ca/root-ca/private').mkdir(
        parents=True, exist_ok=True, mode=0o700)
    (PKI_DIR / 'ca/root-ca/db').mkdir(parents=True, exist_ok=True)

    echo('1.2 Create Database')

    with open(PKI_DIR / 'ca/root-ca/db/root-ca.db', 'w') as f:
        pass  # write empty file
    with open(PKI_DIR / 'ca/root-ca/db/root-ca.db.attr', 'w') as f:
        pass  # write empty file

    with open(PKI_DIR / 'ca/root-ca/db/root-ca.crt.srl', 'w') as f:
        f.write('01')  # write file with '01'
    with open(PKI_DIR / 'ca/root-ca/db/root-ca.crl.srl', 'w') as f:
        f.write('01')  # write file with '01'

    echo('1.3 Create CA Request and Certificate')

    root_crt_path = 'ca/root-ca.crt'
    root_key_path = 'ca/root-ca/private/root-ca.key'

    root_private_key, root_cert = generate_root_cert()

    # Write root CA certificate to disk
    with open(PKI_DIR / root_crt_path, 'wb') as f:
        f.write(root_cert.public_bytes(
            encoding=serialization.Encoding.PEM,
        ))

    with open(PKI_DIR / root_key_path, "wb") as f:
        f.write(root_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    echo('2.  Create Signing Certificate')
    echo('2.1 Create Directories')

    (PKI_DIR / 'ca/signing-ca/private').mkdir(
        parents=True, exist_ok=True, mode=0o700)
    (PKI_DIR / 'ca/signing-ca/db').mkdir(parents=True, exist_ok=True)

    echo('2.2 Create Database')

    with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.db', 'w') as f:
        pass  # write empty file
    with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.db.attr', 'w') as f:
        pass  # write empty file

    with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.crt.srl', 'w') as f:
        f.write('01')  # write file with '01'
    with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.crl.srl', 'w') as f:
        f.write('01')  # write file with '01'

    echo('2.3 Create Signing Certificate CSR')

    signing_csr_path = 'ca/signing-ca.csr'
    signing_crt_path = 'ca/signing-ca.crt'
    signing_key_path = 'ca/signing-ca/private/signing-ca.key'

    signing_private_key, signing_csr = generate_signing_csr()

    # Write Signing CA CSR to disk
    with open(PKI_DIR / signing_csr_path, 'wb') as f:
        f.write(signing_csr.public_bytes(
            encoding=serialization.Encoding.PEM,
        ))

    with open(PKI_DIR / signing_key_path, "wb") as f:
        f.write(signing_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    echo('2.4 Sign Signing Certificate CSR')

    signing_cert = sign_certificate(signing_csr, root_private_key, root_cert.subject, ca=True)

    with open(PKI_DIR / signing_crt_path, 'wb') as f:
        f.write(signing_cert.public_bytes(
            encoding=serialization.Encoding.PEM,
        ))

    echo('3   Create Certificate Chain')

    # create certificate chain file by combining root-ca and signing-ca
    with open(PKI_DIR / 'cert_chain.crt', 'w') as d:
        with open(PKI_DIR / 'ca/root-ca.crt') as s:
            d.write(s.read())
        with open(PKI_DIR / 'ca/signing-ca.crt') as s:
            d.write(s.read())

    echo('\nDone.')


def _get_requirements_dict(txtfile):
    with open(txtfile, 'r') as snapshot:
        snapshot_dict = {}
        for line in snapshot:
            try:
                # 'pip freeze' generates requirements with exact versions
                k, v = line.split('==')
                snapshot_dict[k] = v
            except ValueError:
                snapshot_dict[line] = None
        return snapshot_dict


def _get_dir_hash(path):
    from hashlib import sha256
    hash_ = sha256()
    hash_.update(path.encode('utf-8'))
    hash_ = hash_.hexdigest()
    return hash_


@workspace.command(name='dockerize')
@option('--save',
        required=False,
        help='Save the Docker image into the workspace',
        is_flag=True)
def dockerize_(save):
    """Pack OpenFL and the workspace as a Docker image."""
    import os
    import docker
    import openfl
    from shutil import copy
    from shutil import rmtree
    from os.path import basename

    WORKSPACE_PATH = os.getcwd()
    WORKSPACE_PARENT = Path(WORKSPACE_PATH).parent

    # ~TMP dir
    dirname = ".docker_tmp"
    (SITEPACKS / dirname).mkdir(parents=True, exist_ok=True)

    DOCKER_TMP = os.path.join(SITEPACKS, dirname)

    # Move relevant files into the tmp dir
    # OpenFL BIN files
    # paths definition
    openfl_libs = str(SITEPACKS)
    openfl_bin = get_fx_path(openfl_libs)
    copy(openfl_bin, DOCKER_TMP)

    fx_file = os.path.join(DOCKER_TMP, 'fx')
    replace_line_in_file('#!/usr/local/bin/python3\n', 0, fx_file)

    # Create fl_docker_requirements.txt file
    docker_requirements = "requirements-docker.txt"
    docker_dir = 'openfl-docker'
    copy(os.path.join(SITEPACKS, docker_dir, docker_requirements), DOCKER_TMP)

    # Workspace content
    # copytree(WORKSPACE_PATH,os.path.join(DOCKER_TMP,"workspace"),
    # dirs_exist_ok = True)

    # Docker BUILD COMMAND
    # Define "build_args". These are the equivalent of the "--build-arg"
    # passed to "docker build"
    build_args = {'DOCKER_TMP': dirname, 'openfl_version': openfl.__version__}
    # Add here custom build_args for the build command
    # i.e: build_args["CUSTOM_BUILD_ARG"] = custom_build_arg_var

    # Retrieve args from env vars
    check_varenv('http_proxy', build_args)
    check_varenv('HTTP_PROXY', build_args)
    check_varenv('HTTPS_PROXY', build_args)
    check_varenv('https_proxy', build_args)
    check_varenv('socks_proxy', build_args)
    check_varenv('ftp_proxy', build_args)
    check_varenv('no_proxy', build_args)

    # Compose "build cmd"
    workspace_name = basename(WORKSPACE_PATH)
    openfl_img_base = 'openfl'
    openfl_img_name = f'{openfl_img_base}/docker_{workspace_name}'

    # Clone Dockerfile within SITEPACKS
    docker_dir = "openfl-docker"
    base_df = "Dockerfile.base"
    workspace_df = "Dockerfile.workspace"

    # Copy base dockerfile
    src = os.path.join(SITEPACKS, docker_dir, base_df)
    dst = os.path.join(SITEPACKS, base_df)
    copy(src, dst)

    # Copy workspace dockerfile to directory above workspace
    src = os.path.join(SITEPACKS, docker_dir, workspace_df)
    dst = os.path.join(WORKSPACE_PATH, workspace_df)
    copy(src, dst)

    client = docker.from_env(timeout=3600)

    # Build the image
    try:

        os.chdir(SITEPACKS)
        if f'{openfl_img_base}:latest' in str(client.images.list()):
            if confirm(f'Base image {openfl_img_base} found.'
                       f' Would you like to rebuild?'):
                echo(f'Building docker image {openfl_img_base}. '
                     f'This may take 5-10 minutes')
                client.images.build(
                    path=str(SITEPACKS),
                    tag=openfl_img_base,
                    buildargs=build_args,
                    dockerfile=base_df
                )
        else:
            echo(f'Building docker image {openfl_img_base}.'
                 f' This may take 5-10 minutes')
            client.images.build(
                path=str(SITEPACKS),
                tag=openfl_img_base,
                buildargs=build_args,
                dockerfile=base_df
            )

        echo(f'Building docker image {openfl_img_name}.'
             f' This will likely take 5-10 minutes')
        os.chdir(WORKSPACE_PARENT)
        client.images.build(path=str(WORKSPACE_PATH),
                            tag=openfl_img_name,
                            buildargs=build_args,
                            dockerfile=workspace_df)

    except Exception:
        rmtree(DOCKER_TMP)
        os.remove(os.path.join(SITEPACKS, base_df))
        os.remove(os.path.join(WORKSPACE_PATH, workspace_df))
        raise Exception("Error found while building the image. Aborting!")
        exit()

    echo(f'Docker image {openfl_img_name} successfully built')

    # Clean environment
    rmtree(DOCKER_TMP)
    os.remove(os.path.join(SITEPACKS, base_df))
    os.remove(os.path.join(WORKSPACE_PATH, workspace_df))

    # Produce .tar file containing the freshly built image
    if save:
        archive_fn = f'docker_{workspace_name}.tar'

        os.chdir(WORKSPACE_PATH)
        echo('Saving Docker image...')
        image = client.images.get(f'{openfl_img_name}')
        resp = image.save()
        f = open(archive_fn, 'wb')
        for chunk in resp:
            f.write(chunk)
        f.close()

        echo(f'{openfl_img_name} saved to {WORKSPACE_PATH}/{archive_fn}')
