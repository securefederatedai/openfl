# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This package includes dependencies of the openfl project."""

from setuptools import setup

setup(
    name='openfl',
    version='1.0',
    author='Intel Corporation',
    description='Federated Learning for the Edge',
    packages=[
        'openfl',
        'openfl.interface',
        'openfl.interface.interactive_api',
        'openfl.component',
        'openfl.cryptography',
        'openfl.native',
        'openfl.component.assigner',
        'openfl.component.aggregator',
        'openfl.component.collaborator',
        'openfl.utilities',
        'openfl.protocols',
        'openfl.pipelines',
        'openfl.databases',
        'openfl.transport',
        'openfl.transport.grpc',
        'openfl.federated',
        'openfl.federated.plan',
        'openfl.federated.task',
        'openfl.federated.data',
        'openfl.plugins',
        'openfl.plugins.interface_serializer',
        'openfl.plugins.frameworks_adapters',
        'openfl-workspace',
        'openfl-docker',
        'openfl-tutorials',
    ],
    include_package_data=True,
    install_requires=[
        'Click>=7.0',
        'PyYAML>=5.4.1',
        'numpy',
        'pandas',
        'protobuf',
        'grpcio==1.30.0',
        'grpcio-tools==1.30.0',
        'rich==9.1.0',
        'tqdm',
        'scikit-learn',
        'docker',
        'jupyter',
        'ipykernel',
        'flatten_json',
        'cryptography',
    ],
    python_requires='>=3.6, <3.9',
    entry_points={
        'console_scripts': ['fx=openfl.interface.cli:entry']
    }
)
