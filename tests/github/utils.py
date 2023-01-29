# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from subprocess import check_call
import os
from pathlib import Path
import tarfile


def create_collaborator(col, workspace_root, data_path, archive_name, fed_workspace):
    # Copy workspace to collaborator directories (these can be on different machines)
    col_path = workspace_root / col
    shutil.rmtree(col_path, ignore_errors=True)  # Remove any existing directory
    col_path.mkdir()  # Create a new directory for the collaborator

    # Import the workspace to this collaborator
    check_call(
        ['fx', 'workspace', 'import', '--archive', workspace_root / archive_name],
        cwd=col_path
    )

    # Create collaborator certificate request
    # Remove '--silent' if you run this manually
    check_call(
        ['fx', 'collaborator', 'generate-cert-request', '-d', data_path, '-n', col, '--silent'],
        cwd=col_path / fed_workspace
    )

    # Sign collaborator certificate
    # Remove '--silent' if you run this manually
    request_pkg = col_path / fed_workspace / f'col_{col}_to_agg_cert_request.zip'
    check_call(
        ['fx', 'collaborator', 'certify', '--request-pkg', str(request_pkg), '--silent'],
        cwd=workspace_root)

    # Import the signed certificate from the aggregator
    import_path = workspace_root / f'agg_to_col_{col}_signed_cert.zip'
    check_call(
        ['fx', 'collaborator', 'certify', '--import', import_path],
        cwd=col_path / fed_workspace
    )


def create_certified_workspace(path, template, fqdn, rounds_to_train):
    shutil.rmtree(path, ignore_errors=True)
    check_call(['fx', 'workspace', 'create', '--prefix', path, '--template', template])
    os.chdir(path)

    # Initialize FL plan
    check_call(['fx', 'plan', 'initialize', '-a', fqdn])
    plan_path = Path('plan/plan.yaml')
    edit_plan(plan_path, parameter_dict={'rounds_to_train': rounds_to_train})
    # Create certificate authority for workspace
    check_call(['fx', 'workspace', 'certify'])

    # Export FL workspace
    check_call(['fx', 'workspace', 'export'])


def certify_aggregator(fqdn):
    # Create aggregator certificate
    check_call(['fx', 'aggregator', 'generate-cert-request', '--fqdn', fqdn])

    # Sign aggregator certificate
    check_call(['fx', 'aggregator', 'certify', '--fqdn', fqdn, '--silent'])


def create_signed_cert_for_collaborator(col, data_path) -> Path:
    '''
    We do certs exchage for all participants in a single workspace to speed up this test run.
    Do not do this in real experiments in untrusted environments
    '''
    result_file = Path(f'cert_col_{col}.tar')
    print(f'Certifying collaborator {col} with data path {data_path}...')
    # Create collaborator certificate request
    check_call([
        'fx', 'collaborator', 'generate-cert-request', '-d', data_path, '-n', col, '--silent'
    ])
    # Sign collaborator certificate
    check_call([
        'fx',
        'collaborator',
        'certify',
        '--request-pkg',
        f'col_{col}_to_agg_cert_request.zip',
        '--silent'
    ])

    # Pack the collaborators private key and the signed cert
    # as well as it's data.yaml to a tarball
    tarfiles = ['plan/data.yaml', f'agg_to_col_{col}_signed_cert.zip']
    with os.scandir('cert/client') as iterator:
        for entry in iterator:
            if entry.name.endswith('key'):
                tarfiles.append(entry.path)
    with tarfile.open(result_file, 'w') as t:
        for f in tarfiles:
            t.add(f)
    for f in tarfiles:
        os.remove(f)
    # Remove request archive
    os.remove(f'col_{col}_to_agg_cert_request.zip')

    return result_file.absolute()


def start_aggregator_container(workspace_image_name, aggregator_required_files):
    check_call(
        'docker run --rm '
        '--network host '
        f'-v {Path.cwd().resolve()}/{aggregator_required_files}:/certs.tar '
        '-e \"CONTAINER_TYPE=aggregator\" '
        f'{workspace_image_name} '
        'bash /openfl/openfl-docker/start_actor_in_container.sh',
        shell=True)


def start_collaborator_container(workspace_image_name, col_name):
    check_call(
        'docker run --rm '
        '--network host '
        f'-v {Path.cwd()}/cert_col_{col_name}.tar:/certs.tar '
        '-e \"CONTAINER_TYPE=collaborator\" '
        f'-e \"COL={col_name}\" '
        f'{workspace_image_name} '
        'bash /openfl/openfl-docker/start_actor_in_container.sh',
        shell=True)


def edit_plan(plan_path: Path, parameter_dict: dict) -> None:
    import re
    try:
        with open(plan_path, "r", encoding='utf-8') as sources:
            lines = sources.readlines()
        with open(plan_path, "w", encoding='utf-8') as sources:
            for line in lines:
                for parameter, value in parameter_dict.items():
                    if parameter in line:
                        line = re.sub(
                            f'{parameter}.*',
                            f'{parameter}: {value}',
                            line)
                        break
                sources.write(line)
    except (ValueError, TypeError):
        pass


def edit_plan_yaml(plan_path: Path, parameter_dict: dict) -> None:
    """
    Change plan.yaml settings.

    parameter_dict keys are expected to be dot-delimited,
        for example: {'aggregator.settings.rounds_to_train':5}
    """
    import yaml
    from functools import reduce
    from operator import getitem

    with open(plan_path, "r", encoding='utf-8') as sources:
        plan = yaml.safe_load(sources)
    for parameter, value in parameter_dict.items():
        keys_list = parameter.split('.')
        # Chain dict access
        reduce(getitem, keys_list[:-1], plan)[keys_list[-1]] = value
    with open(plan_path, "w", encoding='utf-8') as sources:
        yaml.safe_dump(plan, sources)
