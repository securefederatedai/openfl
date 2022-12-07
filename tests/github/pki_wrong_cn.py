# Copyright (C) 2020-2021 Intel Corporation
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
            'fx', 'collaborator', 'generate-cert-request',
            '-n', col,
            '-d', '1',
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
    print('Starting Collaborator...')
    col_name = 'one'
    plan = fx.setup_plan()
    plan.resolve()
    client = plan.get_client('two', plan.aggregator_uuid, plan.federation_uuid)
    collaborator = plan.get_collaborator(col_name, client=client)
    collaborator.run()


if __name__ == '__main__':
    origin_dir = os.getcwd()
    prefix = 'fed_workspace'
    subprocess.check_call([
        'fx', 'workspace', 'create',
        '--prefix', prefix,
        '--template', 'keras_cnn_mnist'
    ])
    os.chdir(prefix)
    fqdn = socket.getfqdn()
    prepare_workspace()
    importlib.reload(openfl.federated.task)
    importlib.reload(openfl.federated.data)
    importlib.reload(openfl.federated)
    agg = Process(target=subprocess.check_call, args=[['fx', 'aggregator', 'start']])
    agg.start()
    time.sleep(3)
    failed = False
    try:
        start_invalid_collaborator()
        agg.join()
        assert failed, 'Aggregator accepted invalid collaborator certificate.'
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            failed = True
    finally:
        agg.kill()
