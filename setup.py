# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This package includes dependencies of the openfl project."""

from setuptools import Command
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


class BuildPackageProtos(Command):
    """Command to generate project *_pb2.py modules from proto files."""

    user_options = []

    def initialize_options(self):
        """Set default values for all the options that this command supports.

        Note that these defaults may be overridden by other
        commands, by the setup script, by config files, or by the
        command-line.  Thus, this is not the place to code dependencies
        between options; generally, 'initialize_options()' implementations
        are just a bunch of "self.foo = None" assignments.

        This method must be implemented by all command classes.
        """
        pass

    def finalize_options(self):
        """Set final values for all the options that this command supports.

        This is always called as late as possible, ie.  after any option
        assignments from the command-line or from other commands have been
        done.  Thus, this is the place to code option dependencies: if
        'foo' depends on 'bar', then it is safe to set 'foo' from 'bar' as
        long as 'foo' still has the same value it was assigned in
        'initialize_options()'.

        This method must be implemented by all command classes.
        """
        pass

    def run(self):
        """Build gRPC modules."""
        from grpc.tools import command
        command.build_package_protos('.')


class BuildPyGRPC(build_py):
    """Command for Python modules build."""

    def __init__(self, dist):
        """Create a sub-command to execute."""
        self.subcommand = BuildPackageProtos(dist)
        super().__init__(dist)

    def run(self):
        """Build Python and GRPC modules."""
        self.subcommand.run()
        super().run()


class DevelopGRPC(develop):
    """Command for develop installation."""

    def __init__(self, dist):
        """Create a sub-command to execute."""
        self.subcommand = BuildPackageProtos(dist)
        super().__init__(dist)

    def run(self):
        """Build GRPC modules before the default installation."""
        self.subcommand.run()
        super().run()


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='openfl',
    version='1.5',
    author='Intel Corporation',
    description='Federated Learning for the Edge',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/securefederatedai/openfl',
    packages=[
        'openfl',
        'openfl.component',
        'openfl.interface.aggregation_functions',
        'openfl.interface.aggregation_functions.core',
        'openfl.interface.aggregation_functions.experimental',
        'openfl.component.aggregator',
        'openfl.component.assigner',
        'openfl.component.collaborator',
        'openfl.component.director',
        'openfl.component.envoy',
        'openfl.component.straggler_handling_functions',
        'openfl.cryptography',
        'openfl.databases',
        'openfl.databases.utilities',
        'openfl.experimental',
        'openfl.experimental.workspace_export',
        'openfl.experimental.federated',
        'openfl.experimental.federated.plan',
        'openfl.experimental.component',
        'openfl.experimental.component.aggregator',
        'openfl.experimental.component.collaborator',
        'openfl.experimental.interface.cli',
        'openfl.experimental.interface',
        'openfl.experimental.placement',
        'openfl.experimental.runtime',
        'openfl.experimental.protocols',
        'openfl.experimental.transport',
        'openfl.experimental.transport.grpc',
        'openfl.experimental.utilities',
        'openfl.federated',
        'openfl.federated.data',
        'openfl.federated.plan',
        'openfl.federated.task',
        'openfl.interface',
        'openfl.interface.interactive_api',
        'openfl.native',
        'openfl.pipelines',
        'openfl.plugins',
        'openfl.plugins.frameworks_adapters',
        'openfl.plugins.interface_serializer',
        'openfl.plugins.processing_units_monitor',
        'openfl.protocols',
        'openfl.transport',
        'openfl.transport.grpc',
        'openfl.utilities',
        'openfl.utilities.ca',
        'openfl.utilities.data_splitters',
        'openfl.utilities.fedcurv',
        'openfl.utilities.fedcurv.torch',
        'openfl.utilities.optimizers.keras',
        'openfl.utilities.optimizers.numpy',
        'openfl.utilities.optimizers.torch',
        'openfl-docker',
        'openfl-gramine',
        'openfl-tutorials',
        'openfl-workspace',
    ],
    include_package_data=True,
    install_requires=[
        'Click==8.1.7',
        'PyYAML>=5.4.1',
        'cloudpickle',
        'cryptography>=3.4.6',
        'docker',
        'dynaconf==3.2.5',
        'flatten_json',
        'grpcio>=1.56.2',
        'ipykernel',
        'jupyterlab',
        'numpy',
        'pandas',
        'protobuf>=3.20.3',
        'pyzmq<=26.1.0',
        'requests',
        'rich',
        'scikit-learn',
        'tensorboard',
        'tensorboardX<=2.6',
        'tqdm',
    ],
    setup_requires=['grpcio-tools>=1.56.2,<1.65.0'],
    python_requires='>=3.8, <3.12',
    project_urls={
        'Bug Tracker': 'https://github.com/securefederatedai/openfl/issues',
        'Documentation': 'https://openfl.readthedocs.io/en/stable/',
        'Source Code': 'https://github.com/securefederatedai/openfl',
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',

    ],
    entry_points={
        'console_scripts': ['fx=openfl.interface.cli:entry']
    },
    cmdclass={
        'build_py': BuildPyGRPC,
        'build_grpc': BuildPackageProtos,
        'develop': DevelopGRPC
    },
)
