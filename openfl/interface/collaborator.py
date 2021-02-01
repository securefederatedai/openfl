# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Collaborator module."""

from logging import getLogger
from pathlib import Path
from click import group, option, pass_context
from click import echo, style
from click import Path as ClickPath

from openfl.interface.cli_helper import PKI_DIR
from openfl.federated import Plan

logger = getLogger(__name__)


@group()
@pass_context
def collaborator(context):
    """Manage Federated Learning Collaborators."""
    context.obj['group'] = 'service'


@collaborator.command(name='start')
@pass_context
@option('-p', '--plan', required=False,
        help='Federated learning plan [plan/plan.yaml]',
        default='plan/plan.yaml',
        type=ClickPath(exists=True))
@option('-d', '--data_config', required=False,
        help='The data set/shard configuration file [plan/data.yaml]',
        default='plan/data.yaml', type=ClickPath(exists=True))
@option('-n', '--collaborator_name', required=True,
        help='The certified common name of the collaborator')
@option('-s', '--secure', required=False,
        help='Enable Intel SGX Enclave', is_flag=True, default=False)
def start_(context, plan, collaborator_name, data_config, secure):
    """Start a collaborator service."""
    plan = Plan.Parse(plan_config_path=Path(plan),
                      data_config_path=Path(data_config))

    # TODO: Need to restructure data loader config file loader

    echo(f'Data = {plan.cols_data_paths}')
    logger.info('🧿 Starting a Collaborator Service.')

    plan.get_collaborator(collaborator_name).run()


def RegisterDataPath(collaborator_name, data_path=None, silent=False):
    """Register dataset path in the plan/data.yaml file.

    Args:
        collaborator_name (str): The collaborator whose data path to be defined
        data_path (str)        : Data path (optional)
        silent (bool)          : Silent operation (don't prompt)
    """
    from click import prompt
    from os.path import isfile

    # Ask for the data directory
    default_data_path = f'data/{collaborator_name}'
    if not silent and data_path is None:
        dirPath = prompt('\nWhere is the data (or what is the rank)'
                         ' for collaborator '
                         + style(f'{collaborator_name}', fg='green')
                         + ' ? ', default=default_data_path)
    elif data_path is not None:
        dirPath = data_path
    else:
        # TODO: Need to figure out the default for this.
        dirPath = default_data_path

    # Read the data.yaml file
    d = {}
    data_yaml = 'plan/data.yaml'
    separator = ','
    if isfile(data_yaml):
        with open(data_yaml, 'r') as f:
            for line in f:
                if separator in line:
                    key, val = line.split(separator, maxsplit=1)
                    d[key] = val.strip()

    d[collaborator_name] = dirPath

    # Write the data.yaml
    with open(data_yaml, 'w') as f:
        for key, val in d.items():
            f.write(f'{key}{separator}{val}\n')


@collaborator.command(name='generate-cert-request')
@pass_context
@option('-n', '--collaborator_name', required=True,
        help='The certified common name of the collaborator')
@option('-d', '--data_path',
        help='The data path to be associated with the collaborator')
@option('-s', '--silent', help='Do not prompt', is_flag=True)
@option('-x', '--skip-package',
        help='Do not package the certificate signing request for export',
        is_flag=True)
def generate_cert_request_(context, collaborator_name,
                           data_path, silent, skip_package):
    """Generate certificate request for the collaborator."""
    generate_cert_request(collaborator_name, data_path, silent, skip_package)


def generate_cert_request(collaborator_name, data_path, silent, skip_package):
    """
    Create collaborator certificate key pair.

    Then create a package with the CSR to send for signing.
    """
    from openfl.cryptography.participant import generate_csr
    from openfl.cryptography.io import write_crt, write_key

    common_name = f'{collaborator_name}'.lower()
    subject_alternative_name = f'DNS:{common_name}'
    file_name = f'col_{common_name}'

    echo(f'Creating COLLABORATOR certificate key pair with following settings: '
         f'CN={style(common_name, fg="red")},'
         f' SAN={style(subject_alternative_name, fg="red")}')

    client_private_key, client_csr = generate_csr(common_name, server=False)

    (PKI_DIR / 'client').mkdir(parents=True, exist_ok=True)

    echo('  Moving COLLABORATOR certificate to: ' + style(
        f'{PKI_DIR}/{file_name}', fg='green'))

    # Write collaborator csr and key to disk
    write_crt(client_csr, PKI_DIR / 'client' / f'{file_name}.csr')
    write_key(client_private_key, PKI_DIR / 'client' / f'{file_name}.key')

    if not skip_package:
        from shutil import make_archive, copytree, ignore_patterns
        from tempfile import mkdtemp
        from os.path import join, basename
        from os import remove
        from glob import glob

        archiveType = 'zip'
        archiveName = f'col_{common_name}_to_agg_cert_request'
        archiveFileName = archiveName + '.' + archiveType

        # Collaborator certificate signing request
        tmpDir = join(mkdtemp(), 'openfl', archiveName)

        ignore = ignore_patterns('__pycache__', '*.key', '*.srl', '*.pem')
        # Copy the current directory into the temporary directory
        copytree(f'{PKI_DIR}/client', tmpDir, ignore=ignore)

        for f in glob(f'{tmpDir}/*'):
            if common_name not in basename(f):
                remove(f)

        # Create Zip archive of directory
        make_archive(archiveName, archiveType, tmpDir)

        echo(f'Archive {archiveFileName} with certificate signing'
             f' request created')
        echo('This file should be sent to the certificate authority'
             ' (typically hosted by the aggregator) for signing')

    # TODO: There should be some association with the plan made here as well
    RegisterDataPath(common_name, data_path=data_path, silent=silent)


def findCertificateName(file_name):
    """Parse the collaborator name."""
    col_name = str(file_name).split('/')[-1].split('.')[0][4:]
    return col_name


def RegisterCollaborator(file_name):
    """Register the collaborator name in the cols.yaml list.

    Args:
        file_name (str): The name of the collaborator in this federation

    """
    from yaml import load, dump, FullLoader

    col_name = findCertificateName(file_name)

    cols_file = 'plan/cols.yaml'

    with open(cols_file, 'r') as f:
        doc = load(f, Loader=FullLoader)

    if not doc:  # YAML is not correctly formatted
        doc = {}  # Create empty dictionary

    # List doesn't exist
    if 'collaborators' not in doc.keys() or not doc['collaborators']:
        doc['collaborators'] = []  # Create empty list

    if col_name in doc['collaborators']:

        echo('\nCollaborator '
             + style(f'{col_name}', fg='green')
             + ' is already in the '
             + style(f'{cols_file}', fg='green'))

    else:

        doc['collaborators'].append(col_name)
        with open(cols_file, 'w') as f:
            dump(doc, f)

        echo('\nRegistering '
             + style(f'{col_name}', fg='green')
             + ' in '
             + style(f'{cols_file}', fg='green'))


@collaborator.command(name='certify')
@pass_context
@option('-n', '--collaborator_name',
        help='The certified common name of the collaborator. This is only'
             ' needed for single node expiriments')
@option('-s', '--silent', help='Do not prompt', is_flag=True)
@option('-r', '--request-pkg',
        help='The archive containing the certificate signing'
             ' request (*.zip) for a collaborator')
@option('-i', '--import', 'import_',
        help='Import the archive containing the collaborator\'s'
             ' certificate (signed by the CA)')
def certify_(context, collaborator_name, silent, request_pkg, import_):
    """Certify the collaborator."""
    certify(collaborator_name, silent, request_pkg, import_)


def certify(collaborator_name, silent, request_pkg=False, import_=False):
    """Sign/certify collaborator certificate key pair."""
    from click import confirm

    from shutil import unpack_archive
    from shutil import make_archive, copy
    from glob import glob
    from os.path import basename, join, splitext
    from os import remove
    from tempfile import mkdtemp
    from openfl.cryptography.ca import sign_certificate
    from openfl.cryptography.io import read_key, read_crt, read_csr
    from openfl.cryptography.io import write_crt

    common_name = f'{collaborator_name}'.lower()

    if not import_:
        if request_pkg:
            Path(f'{PKI_DIR}/client').mkdir(parents=True, exist_ok=True)
            unpack_archive(request_pkg, extract_dir=f'{PKI_DIR}/client')
            csr = glob(f'{PKI_DIR}/client/*.csr')[0]
        else:
            if collaborator_name is None:
                echo('collaborator_name can only be omitted if signing\n'
                     'a zipped request package.\n'
                     '\n'
                     'Example: fx collaborator certify --request-pkg '
                     'col_one_to_agg_cert_request.zip')
                return
            csr = glob(f'{PKI_DIR}/client/col_{common_name}.csr')[0]
            copy(csr, PKI_DIR)
        cert_name = splitext(csr)[0]
        file_name = basename(cert_name)
        signing_key_path = 'ca/signing-ca/private/signing-ca.key'
        signing_crt_path = 'ca/signing-ca.crt'

        # Load CSR
        if not Path(f'{cert_name}.csr').exists():
            echo(style('Collaborator certificate signing request not found.', fg='red')
                 + ' Please run `fx collaborator generate-cert-request`'
                 ' to generate the certificate request.')

        csr, csr_hash = read_csr(f'{cert_name}.csr')

        # Load private signing key
        if not Path(PKI_DIR / signing_key_path).exists():
            echo(style('Signing key not found.', fg='red')
                 + ' Please run `fx workspace certify`'
                 ' to initialize the local certificate authority.')

        signing_key = read_key(PKI_DIR / signing_key_path)

        # Load signing cert
        if not Path(PKI_DIR / signing_crt_path).exists():
            echo(style('Signing certificate not found.', fg='red')
                 + ' Please run `fx workspace certify`'
                 ' to initialize the local certificate authority.')

        signing_crt = read_crt(PKI_DIR / signing_crt_path)

        echo('The CSR Hash for file '
             + style(f'{file_name}.csr', fg='green')
             + ' = '
             + style(f'{csr_hash}', fg='red'))

        if silent:

            echo(' Signing COLLABORATOR certificate')
            signed_col_cert = sign_certificate(csr, signing_key, signing_crt.subject)
            write_crt(signed_col_cert, f'{cert_name}.crt')
            RegisterCollaborator(PKI_DIR / 'client' / f'{file_name}.crt')

        else:

            if confirm("Do you want to sign this certificate?"):

                echo(' Signing COLLABORATOR certificate')
                signed_col_cert = sign_certificate(csr, signing_key, signing_crt.subject)
                write_crt(signed_col_cert, f'{cert_name}.crt')
                RegisterCollaborator(PKI_DIR / 'client' / f'{file_name}.crt')

            else:
                echo(style('Not signing certificate.', fg='red')
                     + ' Please check with this collaborator to get the'
                       ' correct certificate for this federation.')
                return

        if len(common_name) == 0:
            # If the collaborator name is provided, the collaborator and
            # certificate does not need to be exported
            return

        # Remove unneeded CSR
        remove(f'{cert_name}.csr')

        archiveType = 'zip'
        archiveName = f'agg_to_{file_name}_signed_cert'
        # archiveFileName = archiveName + '.' + archiveType

        # Collaborator certificate signing request
        tmpDir = join(mkdtemp(), 'openfl', archiveName)

        Path(f'{tmpDir}/client').mkdir(parents=True, exist_ok=True)
        # Copy the signed cert to the temporary directory
        copy(f'{PKI_DIR}/client/{file_name}.crt', f'{tmpDir}/client/')
        # Copy the CA certificate chain to the temporary directory
        copy(f'{PKI_DIR}/cert_chain.crt', tmpDir)

        # Create Zip archive of directory
        make_archive(archiveName, archiveType, tmpDir)

    else:
        # Copy the signed certificate and cert chain into PKI_DIR
        previous_crts = glob(f'{PKI_DIR}/client/*.crt')
        unpack_archive(import_, extract_dir=PKI_DIR)
        updated_crts = glob(f'{PKI_DIR}/client/*.crt')
        cert_difference = list(set(updated_crts) - set(previous_crts))
        if len(cert_difference) == 0:
            crt = basename(cert_difference[0])
            echo(f"Certificate {crt} installed to PKI directory")
        else:
            crt = basename(updated_crts[0])
            echo("Certificate updated in the PKI directory")
