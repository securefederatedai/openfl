# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import grpc
import subprocess
import os
import time
import socket
from multiprocessing import Process
import sys
import importlib

import openfl
import openfl.native as fx
from openfl.utilities.utils import getfqdn_env

def prepare_workspace():
    subprocess.check_call(['fx', 'workspace', 'certify'])
    subprocess.check_call(['fx', 'plan', 'initialize'])

    subprocess.check_call([
        'fx', 'aggregator', 'generate-cert-request'
    ])
    subprocess.check_call([
        'fx', 'aggregator', 'certify',
        '-s'
    ])
    for col in ['one', 'two']:
        subprocess.check_call([
            'fx', 'collaborator', 'create',
            '-n', col,
            '-d', '1',
            '-s'
        ])
        subprocess.check_call([
            'fx', 'collaborator', 'generate-cert-request',
            '-n', col,
            '-s', '-x'
        ])
        subprocess.check_call([
            'fx', 'collaborator', 'certify',
            '-n', col,
            '-s'
        ])

    sys.path.append(os.getcwd())


def start_invalid_collaborator():
    '''
    We choose the gRPC client of another collaborator
    to check if aggregator accepts certificate
    that does not correspond to the collaborator's name.
    '''
    importlib.reload(openfl.federated.task)  # fetch TF-based task runner
    importlib.reload(openfl.federated.data)  # fetch TF-based data loader
    importlib.reload(openfl.federated)  # allow imports from parent module
    col_name = 'one'
    plan = fx.setup_plan()
    plan.resolve()
    client = plan.get_client('two', plan.aggregator_uuid, plan.federation_uuid)
    collaborator = plan.get_collaborator(col_name, client=client)
    collaborator.run()


def start_aggregator():
    agg = Process(target=subprocess.check_call, args=[['fx', 'aggregator', 'start']])
    agg.start()
    time.sleep(3)  # wait for initialization
    return agg


if __name__ == '__main__':
    origin_dir = os.getcwd()
    prefix = 'fed_workspace'
    subprocess.check_call([
        'fx', 'workspace', 'create',
        '--prefix', prefix,
        '--template', 'keras_cnn_mnist'
    ])
    os.chdir(prefix)
    fqdn = getfqdn_env()
    prepare_workspace()
    agg = start_aggregator()
    try:
        start_invalid_collaborator()
        agg.join()
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            pass
        else:
            raise
    else:
        print('Aggregator accepted invalid collaborator certificate.')
        sys.exit(1)
    finally:
        agg.kill()
