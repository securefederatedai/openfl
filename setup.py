# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This package includes dependencies of the openfl project."""

from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='openfl',
    version='1.5',
    author='Intel Corporation',
    description='Federated Learning for the Edge',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/intel/openfl',
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
        'openfl.utilities.optimizers.torch',
        'openfl.utilities.optimizers.keras',
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
        'openfl.component.aggregation_functions'
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
        'cryptography>=3.4.6',
        'cloudpickle',
    ],
    python_requires='>=3.6, <3.9',
    project_urls={
        "Bug Tracker": "https://github.com/intel/openfl/issues",
        "Documentation": "https://openfl.readthedocs.io/en/stable/",
        "Source Code": "https://github.com/intel/openfl",
    },
    classifiers=[
        'Environment :: Console',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 5 - Production/Stable',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: System :: Distributed Computing',
        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={
        'console_scripts': ['fx=openfl.interface.cli:entry']
    }
)
