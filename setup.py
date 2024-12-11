# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Setup script."""

from setuptools import Command, find_packages, setup
from setuptools.command.build_py import build_py


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


setup(
    name='openfl',
    version='1.6',
    author='OpenFL Team',
    description='Federated Learning for the Edge',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/securefederatedai/openfl',
    packages=find_packages(
        include=(
            "openfl",
            "openfl.*",
            "openfl-docker",
            "openfl-workspace",
            "openfl-tutorials",
        )
    ),
    include_package_data=True,
    setup_requires=['grpcio-tools>=1.56.2,<1.66.0'],  # ensure it is in-sync with `install_requires`
    install_requires=[
        'click',
        'psutil',
        'pyyaml',
        'rich',
        'dynaconf',
        'tqdm',
        'numpy',
        'requests>=2.32.0',
        'cloudpickle',
        'cryptography',
        'pandas',
        'scikit-learn',
        'flatten_json',
        'tensorboardX',
        'protobuf>=4.22,<6.0.0',
        'grpcio>=1.56.2,<1.66.0',
    ],
    python_requires='>=3.9, <3.12',
    project_urls={
        'Bug Tracker': 'https://github.com/securefederatedai/openfl/issues',
        'Documentation': 'https://openfl.readthedocs.io/en/stable/',
        'Source Code': 'https://github.com/securefederatedai/openfl',
    },
    classifiers=[
        'Environment :: Console',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Distributed Computing',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={'console_scripts': ['fx=openfl.interface.cli:entry']},
    cmdclass={
        'build_py': BuildPyGRPC,
        'build_grpc': BuildPackageProtos,
    },
)
