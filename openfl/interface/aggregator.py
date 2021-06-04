# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Aggregator module."""

from socket import getfqdn
from logging import getLogger
from pathlib import Path
from click import Path as ClickPath
from click import group, option, pass_context
from click import echo, style

from openfl.federated import Plan
from openfl.interface.cli_helper import PKI_DIR


logger = getLogger(__name__)

step = './step/step_0.15.16/bin/step'
step_ca = './step/step-ca_0.15.15/bin/step-ca'

@group()
@pass_context
def aggregator(context):
    """Manage Federated Learning Aggregator."""
    context.obj['group'] = 'aggregator'


@aggregator.command(name='start')
@pass_context
@option('-p', '--plan', required=False,
        help='Federated learning plan [plan/plan.yaml]',
        default='plan/plan.yaml',
        type=ClickPath(exists=True))
@option('-c', '--authorized_cols', required=False,
        help='Authorized collaborator list [plan/cols.yaml]',
        default='plan/cols.yaml', type=ClickPath(exists=True))
@option('-s', '--secure', required=False,
        help='Enable Intel SGX Enclave', is_flag=True, default=False)
def start_(context, plan, authorized_cols, secure):
    """Start the aggregator service."""
    plan = Plan.Parse(plan_config_path=Path(plan),
                      cols_config_path=Path(authorized_cols))

    logger.info('🧿 Starting the Aggregator Service.')

    plan.get_server().serve()


# @aggregator.command(name='generate-cert-request')
# @option('--fqdn', required=False,
#         help=f'The fully qualified domain name of'
#              f' aggregator node [{getfqdn()}]',
#         default=getfqdn())
# def _generate_cert_request(fqdn):
#     generate_cert_request(fqdn)


@aggregator.command(name='certify')
@option('-n', '--fqdn',
        help='The fully qualified domain name of aggregator node [{getfqdn()}]',
        default=None)
@option('-s', '--silent', help='Do not prompt', is_flag=True)
@option('-t', '--token', help='Do not prompt', required=True)
def _certify(fqdn, silent, token):
    certify(fqdn, silent, token)

def certify(fqdn, silent, message):
    """Create aggregator certificate key pair."""
    from openfl.cryptography.participant import generate_csr
    from openfl.cryptography.io import write_crt, write_key

    if fqdn is None:
        fqdn = 'nnlicv674.inn.intel.com'
    print('certify', fqdn)
    # fqdn = 'localhost'
    import os
    import base64
    length = int(message[:4])
    print(length)
    token = message[4:length +4]
    root_ca = message[length+4:]
#    base64_bytes = root_ca.encode('ascii')
    message_bytes = base64.b64decode(root_ca)

    with open('./cert/root_ca.crt',mode='wb') as file:
        file.write(message_bytes)
    # root_ca = message_bytes.decode('ascii')
    # print(root_ca)

    os.system(f'./{step} ca certificate {fqdn} agg_{fqdn}.crt agg_{fqdn}.key -f --token {token}')
    # common_name = f'{fqdn}'.lower()
    # subject_alternative_name = f'DNS:{common_name}'
    # file_name = f'agg_{common_name}'

    # echo(f'Creating AGGREGATOR certificate key pair with following settings: '
    #      f'CN={style(common_name, fg="red")},'
    #      f' SAN={style(subject_alternative_name, fg="red")}')

    # server_private_key, server_csr = generate_csr(common_name, server=True)

    # (PKI_DIR / 'server').mkdir(parents=True, exist_ok=True)

    # echo('  Writing AGGREGATOR certificate key pair to: ' + style(
    #     f'{PKI_DIR}/server', fg='green'))

    # # Write aggregator csr and key to disk
    # write_crt(server_csr, PKI_DIR / 'server' / f'{file_name}.csr')
    # write_key(server_private_key, PKI_DIR / 'server' / f'{file_name}.key')


def findCertificateName(file_name):
    """Search the CRT for the actual aggregator name."""
    # This loop looks for the collaborator name in the key
    with open(file_name, 'r') as f:
        for line in f:
            if 'Subject: CN=' in line:
                col_name = line.split('=')[-1].strip()
                break
    return col_name


# @aggregator.command(name='certify')
# @option('-n', '--fqdn',
#         help='The fully qualified domain name of aggregator node [{getfqdn()}]',
#         default=getfqdn())
# @option('-s', '--silent', help='Do not prompt', is_flag=True)
# def _certify(fqdn, silent):
#     certify(fqdn, silent)


# def certify(fqdn, silent):
#     """Sign/certify the aggregator certificate key pair."""
#     from openfl.cryptography.ca import sign_certificate
#     from openfl.cryptography.io import read_key, read_crt, read_csr
#     from openfl.cryptography.io import write_crt

#     from click import confirm

#     if fqdn is None:
#         fqdn = getfqdn()

#     common_name = f'{fqdn}'.lower()
#     file_name = f'agg_{common_name}'
#     cert_name = f'server/{file_name}'
#     signing_key_path = 'ca/signing-ca/private/signing-ca.key'
#     signing_crt_path = 'ca/signing-ca.crt'

#     # Load CSR
#     if not Path(PKI_DIR / f'{cert_name}.csr').exists():
#         echo(style('Aggregator certificate signing request not found.', fg='red')
#              + ' Please run `fx aggregator generate-cert-request`'
#              ' to generate the certificate request.')

#     csr, csr_hash = read_csr(PKI_DIR / f'{cert_name}.csr')

#     # Load private signing key
#     if not Path(PKI_DIR / signing_key_path).exists():
#         echo(style('Signing key not found.', fg='red')
#              + ' Please run `fx workspace certify`'
#              ' to initialize the local certificate authority.')

#     signing_key = read_key(PKI_DIR / signing_key_path)

#     # Load signing cert
#     if not Path(PKI_DIR / signing_crt_path).exists():
#         echo(style('Signing certificate not found.', fg='red')
#              + ' Please run `fx workspace certify`'
#              ' to initialize the local certificate authority.')

#     signing_crt = read_crt(PKI_DIR / signing_crt_path)

#     echo('The CSR Hash for file '
#          + style(f'{cert_name}.csr', fg='green')
#          + ' = '
#          + style(f'{csr_hash}', fg='red'))

#     if silent:

#         echo(' Signing AGGREGATOR certificate')
#         signed_agg_cert = sign_certificate(csr, signing_key, signing_crt.subject)
#         write_crt(signed_agg_cert, PKI_DIR / f'{cert_name}.crt')

#     else:

#         if confirm("Do you want to sign this certificate?"):

#             echo(' Signing AGGREGATOR certificate')
#             signed_agg_cert = sign_certificate(csr, signing_key, signing_crt.subject)
#             write_crt(signed_agg_cert, PKI_DIR / f'{cert_name}.crt')

#         else:
#             echo(style('Not signing certificate.', fg='red')
#                  + ' Please check with this AGGREGATOR to get the correct'
#                    ' certificate for this federation.')
