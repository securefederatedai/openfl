# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Workspace module."""

import sys
from pathlib import Path

from click import Choice
from click import confirm
from click import echo
from click import group
from click import option
from click import pass_context
from click import Path as ClickPath

from openfl.utilities.path_check import is_directory_traversal


@group()
@pass_context
def workspace(context):
    """Manage Federated Learning Workspaces."""
    context.obj['group'] = 'workspace'


def create_dirs(prefix):
    """Create workspace directories."""
    from shutil import copyfile

    from openfl.interface.cli_helper import WORKSPACE

    echo('Creating Workspace Directories')

    (prefix / 'cert').mkdir(parents=True, exist_ok=True)  # certifications
    (prefix / 'data').mkdir(parents=True, exist_ok=True)  # training data
    (prefix / 'logs').mkdir(parents=True, exist_ok=True)  # training logs
    (prefix / 'save').mkdir(parents=True, exist_ok=True)  # model weight saves / initialization
    (prefix / 'src').mkdir(parents=True, exist_ok=True)  # model code

    copyfile(WORKSPACE / 'workspace' / '.workspace', prefix / '.workspace')


def create_temp(prefix, template):
    """Create workspace templates."""
    from shutil import ignore_patterns

    from openfl.interface.cli_helper import copytree
    from openfl.interface.cli_helper import WORKSPACE

    echo('Creating Workspace Templates')

    copytree(src=WORKSPACE / template, dst=prefix, dirs_exist_ok=True,
             ignore=ignore_patterns('__pycache__'))  # from template workspace
    apply_template_plan(prefix, template)


def get_templates():
    """Grab the default templates from the distribution."""
    from openfl.interface.cli_helper import WORKSPACE

    return [d.name for d in WORKSPACE.glob('*') if d.is_dir()
            and d.name not in ['__pycache__', 'workspace']]


@workspace.command(name='create')
@option('--prefix', required=True,
        help='Workspace name or path', type=ClickPath())
@option('--template', required=True, type=Choice(get_templates()))
def create_(prefix, template):
    """Create the workspace."""
    if is_directory_traversal(prefix):
        echo('Workspace name or path is out of the openfl workspace scope.')
        sys.exit(1)
    create(prefix, template)


def create(prefix, template):
    """Create federated learning workspace."""
    from os.path import isfile
    from subprocess import check_call
    from sys import executable

    from openfl.interface.cli_helper import print_tree
    from openfl.interface.cli_helper import OPENFL_USERDIR

    if not OPENFL_USERDIR.exists():
        OPENFL_USERDIR.mkdir()

    prefix = Path(prefix).absolute()

    create_dirs(prefix)
    create_temp(prefix, template)

    requirements_filename = 'requirements.txt'

    if isfile(f'{str(prefix)}/{requirements_filename}'):
        check_call([
            executable, '-m', 'pip', 'install', '-r',
            f'{prefix}/requirements.txt'], shell=False)
        echo(f'Successfully installed packages from {prefix}/requirements.txt.')
    else:
        echo('No additional requirements for workspace defined. Skipping...')
    prefix_hash = _get_dir_hash(str(prefix.absolute()))
    with open(OPENFL_USERDIR / f'requirements.{prefix_hash}.txt', 'w') as f:
        check_call([executable, '-m', 'pip', 'freeze'], shell=False, stdout=f)

    print_tree(prefix, level=3)


@workspace.command(name='export')
def export_():
    """Export federated learning workspace."""
    from os import getcwd
    from os import makedirs
    from os.path import basename
    from os.path import join
    from shutil import copy2
    from shutil import copytree
    from shutil import ignore_patterns
    from shutil import make_archive
    from tempfile import mkdtemp

    from plan import freeze_plan
    from openfl.interface.cli_helper import WORKSPACE

    # TODO: Does this need to freeze all plans?
    plan_file = Path('plan/plan.yaml').absolute()
    try:
        freeze_plan(plan_file)
    except Exception:
        echo(f'Plan file "{plan_file}" not found. No freeze performed.')

    from pip._internal.operations import freeze
    requirements_generator = freeze.freeze()
    with open('./requirements.txt', 'w') as f:
        for package in requirements_generator:
            if '==' not in package:
                # We do not export dependencies without version
                continue
            f.write(package + '\n')

    archive_type = 'zip'
    archive_name = basename(getcwd())
    archive_file_name = archive_name + '.' + archive_type

    # Aggregator workspace
    tmp_dir = join(mkdtemp(), 'openfl', archive_name)

    ignore = ignore_patterns(
        '__pycache__', '*.crt', '*.key', '*.csr', '*.srl', '*.pem', '*.pbuf')

    # We only export the minimum required files to set up a collaborator
    makedirs(f'{tmp_dir}/save', exist_ok=True)
    makedirs(f'{tmp_dir}/logs', exist_ok=True)
    makedirs(f'{tmp_dir}/data', exist_ok=True)
    copytree('./src', f'{tmp_dir}/src', ignore=ignore)  # code
    copytree('./plan', f'{tmp_dir}/plan', ignore=ignore)  # plan
    copy2('./requirements.txt', f'{tmp_dir}/requirements.txt')  # requirements

    try:
        copy2('.workspace', tmp_dir)  # .workspace
    except FileNotFoundError:
        echo('\'.workspace\' file not found.')
        if confirm('Create a default \'.workspace\' file?'):
            copy2(WORKSPACE / 'workspace' / '.workspace', tmp_dir)
        else:
            echo('To proceed, you must have a \'.workspace\' '
                 'file in the current directory.')
            raise

    # Create Zip archive of directory
    make_archive(archive_name, archive_type, tmp_dir)

    echo(f'Workspace exported to archive: {archive_file_name}')


@workspace.command(name='import')
@option('--archive', required=True,
        help='Zip file containing workspace to import',
        type=ClickPath(exists=True))
def import_(archive):
    """Import federated learning workspace."""
    from os import chdir
    from os.path import basename
    from os.path import isfile
    from shutil import unpack_archive
    from subprocess import check_call
    from sys import executable

    if is_directory_traversal(archive):
        echo('Archive path is out of the openfl workspace scope.')
        sys.exit(1)

    archive = Path(archive).absolute()

    dir_path = basename(archive).split('.')[0]
    unpack_archive(archive, extract_dir=dir_path)
    chdir(dir_path)

    requirements_filename = 'requirements.txt'

    if isfile(requirements_filename):
        check_call([
            executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            shell=False)
    else:
        echo('No ' + requirements_filename + ' file found.')

    echo(f'Workspace {archive} has been imported.')
    echo('You may need to copy your PKI certificates to join the federation.')


@workspace.command(name='certify')
def certify_():
    """Create certificate authority for federation."""
    certify()


def certify():
    """Create certificate authority for federation."""
    from cryptography.hazmat.primitives import serialization

    from openfl.cryptography.ca import generate_root_cert
    from openfl.cryptography.ca import generate_signing_csr
    from openfl.cryptography.ca import sign_certificate
    from openfl.interface.cli_helper import CERT_DIR

    echo('Setting Up Certificate Authority...\n')

    echo('1.  Create Root CA')
    echo('1.1 Create Directories')

    (CERT_DIR / 'ca/root-ca/private').mkdir(
        parents=True, exist_ok=True, mode=0o700)
    (CERT_DIR / 'ca/root-ca/db').mkdir(parents=True, exist_ok=True)

    echo('1.2 Create Database')

    with open(CERT_DIR / 'ca/root-ca/db/root-ca.db', 'w') as f:
        pass  # write empty file
    with open(CERT_DIR / 'ca/root-ca/db/root-ca.db.attr', 'w') as f:
        pass  # write empty file

    with open(CERT_DIR / 'ca/root-ca/db/root-ca.crt.srl', 'w') as f:
        f.write('01')  # write file with '01'
    with open(CERT_DIR / 'ca/root-ca/db/root-ca.crl.srl', 'w') as f:
        f.write('01')  # write file with '01'

    echo('1.3 Create CA Request and Certificate')

    root_crt_path = 'ca/root-ca.crt'
    root_key_path = 'ca/root-ca/private/root-ca.key'

    root_private_key, root_cert = generate_root_cert()

    # Write root CA certificate to disk
    with open(CERT_DIR / root_crt_path, 'wb') as f:
        f.write(root_cert.public_bytes(
            encoding=serialization.Encoding.PEM,
        ))

    with open(CERT_DIR / root_key_path, 'wb') as f:
        f.write(root_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    echo('2.  Create Signing Certificate')
    echo('2.1 Create Directories')

    (CERT_DIR / 'ca/signing-ca/private').mkdir(
        parents=True, exist_ok=True, mode=0o700)
    (CERT_DIR / 'ca/signing-ca/db').mkdir(parents=True, exist_ok=True)

    echo('2.2 Create Database')

    with open(CERT_DIR / 'ca/signing-ca/db/signing-ca.db', 'w') as f:
        pass  # write empty file
    with open(CERT_DIR / 'ca/signing-ca/db/signing-ca.db.attr', 'w') as f:
        pass  # write empty file

    with open(CERT_DIR / 'ca/signing-ca/db/signing-ca.crt.srl', 'w') as f:
        f.write('01')  # write file with '01'
    with open(CERT_DIR / 'ca/signing-ca/db/signing-ca.crl.srl', 'w') as f:
        f.write('01')  # write file with '01'

    echo('2.3 Create Signing Certificate CSR')

    signing_csr_path = 'ca/signing-ca.csr'
    signing_crt_path = 'ca/signing-ca.crt'
    signing_key_path = 'ca/signing-ca/private/signing-ca.key'

    signing_private_key, signing_csr = generate_signing_csr()

    # Write Signing CA CSR to disk
    with open(CERT_DIR / signing_csr_path, 'wb') as f:
        f.write(signing_csr.public_bytes(
            encoding=serialization.Encoding.PEM,
        ))

    with open(CERT_DIR / signing_key_path, 'wb') as f:
        f.write(signing_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    echo('2.4 Sign Signing Certificate CSR')

    signing_cert = sign_certificate(signing_csr, root_private_key, root_cert.subject, ca=True)

    with open(CERT_DIR / signing_crt_path, 'wb') as f:
        f.write(signing_cert.public_bytes(
            encoding=serialization.Encoding.PEM,
        ))

    echo('3   Create Certificate Chain')

    # create certificate chain file by combining root-ca and signing-ca
    with open(CERT_DIR / 'cert_chain.crt', 'w') as d:
        with open(CERT_DIR / 'ca/root-ca.crt') as s:
            d.write(s.read())
        with open(CERT_DIR / 'ca/signing-ca.crt') as s:
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
@option('-b', '--base_image', required=False,
        help='The tag for openfl base image',
        default='openfl')
@option('--save/--no-save',
        required=False,
        help='Save the Docker image into the workspace',
        default=True)
@pass_context
def dockerize_(context, base_image, save):
    """
    Pack the workspace as a Docker image.

    This command is the alternative to `workspace export`.
    It should be called after plan initialization from the workspace dir.

    User is expected to be in docker group.
    If your machine is behind a proxy, make sure you set it up in ~/.docker/config.json.
    """
    import docker
    import os
    import sys
    from shutil import copyfile

    from openfl.interface.cli_helper import SITEPACKS

    # Specify the Dockerfile.workspace loaction
    openfl_docker_dir = os.path.join(SITEPACKS, 'openfl-docker')
    dockerfile_workspace = 'Dockerfile.workspace'
    # Apparently, docker's python package does not support
    # scenarios when the dockerfile is placed outside the build context
    copyfile(os.path.join(openfl_docker_dir, dockerfile_workspace), dockerfile_workspace)

    workspace_path = os.getcwd()
    workspace_name = os.path.basename(workspace_path)

    # Exporting the workspace
    context.invoke(export_)
    workspace_archive = workspace_name + '.zip'

    build_args = {
        'WORKSPACE_NAME': workspace_name,
        'BASE_IMAGE': base_image
    }

    client = docker.from_env(timeout=3600)
    try:
        echo('Building the Docker image')
        client.images.build(
            path=str(workspace_path),
            tag=workspace_name,
            buildargs=build_args,
            dockerfile=dockerfile_workspace)

    except Exception as e:
        echo('Failed to build the image\n' + str(e) + '\n')
        sys.exit(1)
    else:
        echo('The workspace image has been built successfully!')
    finally:
        os.remove(workspace_archive)
        os.remove(dockerfile_workspace)

    # Saving the image to a tarball
    if save:
        workspace_image_tar = workspace_name + '_image.tar'
        echo('Saving the Docker image...')
        image = client.images.get(f'{workspace_name}')
        resp = image.save(named=True)
        with open(workspace_image_tar, 'wb') as f:
            for chunk in resp:
                f.write(chunk)
        echo(f'{workspace_name} image saved to {workspace_path}/{workspace_image_tar}')


def apply_template_plan(prefix, template):
    """Copy plan file from template folder.

    This function unfolds default values from template plan configuration
    and writes the configuration to the current workspace.
    """
    from openfl.federated.plan import Plan
    from openfl.interface.cli_helper import WORKSPACE

    template_plan = Plan.parse(WORKSPACE / template / 'plan' / 'plan.yaml')

    Plan.dump(prefix / 'plan' / 'plan.yaml', template_plan.config)
