# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Workspace utils module."""

import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from subprocess import check_call
from sys import executable
from typing import Optional
from typing import Tuple
from typing import Union

logger = logging.getLogger(__name__)


class ExperimentWorkspace:
    """Experiment workspace context manager."""

    def __init__(
            self,
            experiment_name: str,
            data_file_path: Path,
            install_requirements: bool = False,
            remove_archive: bool = True
    ) -> None:
        """Initialize workspace context manager."""
        self.experiment_name = experiment_name
        self.data_file_path = data_file_path
        self.install_requirements = install_requirements
        self.cwd = Path.cwd()
        self.experiment_work_dir = self.cwd / self.experiment_name
        self.remove_archive = remove_archive

    def _install_requirements(self):
        """Install experiment requirements."""
        requirements_filename = self.experiment_work_dir / 'requirements.txt'

        if requirements_filename.is_file():
            attempts = 10
            for _ in range(attempts):
                try:
                    check_call([
                        executable, '-m', 'pip', 'install', '-r', requirements_filename],
                        shell=False)
                except Exception as exc:
                    logger.error(f'Failed to install requirements: {exc}')
                    # It's a workaround for cases when collaborators run
                    # in common virtual environment
                    time.sleep(5)
                else:
                    break
        else:
            logger.error('No ' + requirements_filename + ' file found.')

    def __enter__(self):
        """Create a collaborator workspace for the experiment."""
        if self.experiment_work_dir.exists():
            shutil.rmtree(self.experiment_work_dir, ignore_errors=True)
        os.makedirs(self.experiment_work_dir)

        shutil.unpack_archive(self.data_file_path, self.experiment_work_dir, format='zip')

        if self.install_requirements:
            self._install_requirements()

        os.chdir(self.experiment_work_dir)

        # This is needed for python module finder
        sys.path.append(str(self.experiment_work_dir))

    def __exit__(self, exc_type, exc_value, traceback):
        """Remove the workspace."""
        os.chdir(self.cwd)
        shutil.rmtree(self.experiment_work_dir, ignore_errors=True)
        if str(self.experiment_work_dir) in sys.path:
            sys.path.remove(str(self.experiment_work_dir))

        if self.remove_archive:
            logger.debug(
                'Exiting from the workspace context manager'
                f' for {self.experiment_name} experiment'
            )
            logger.debug(f'Archive still exists: {self.data_file_path.exists()}')
            self.data_file_path.unlink(missing_ok=False)


def dump_requirements_file(
        path: Union[str, Path] = './requirements.txt',
        keep_original_prefixes: bool = True,
        prefixes: Optional[Union[Tuple[str], str]] = None,
) -> None:
    """Prepare and save requirements.txt."""
    from pip._internal.operations import freeze
    path = Path(path).absolute()

    # Prepare user provided prefixes for merge with original ones
    if prefixes is None:
        prefixes = set()
    elif type(prefixes) is str:
        prefixes = set(prefixes,)
    else:
        prefixes = set(prefixes)

    # Merge prefixes:
    # We expect that all the prefixes in a requirement file
    # are placed at the top
    if keep_original_prefixes and path.is_file():
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    continue
                if line[0] == '-':
                    prefixes |= {line.replace('\n', '')}
                else:
                    break

    requirements_generator = freeze.freeze()
    with open(path, 'w', encoding='utf-8') as f:
        for prefix in prefixes:
            f.write(prefix + '\n')

        for package in requirements_generator:
            if _is_package_versioned(package):
                f.write(package + '\n')


def _is_package_versioned(package: str) -> bool:
    """Check if the package has a version."""
    return ('==' in package
            and package not in ['pkg-resources==0.0.0', 'pkg_resources==0.0.0']
            and '-e ' not in package
            )


@contextmanager
def set_directory(path: Path):
    """
    Sets provided path as the cwd within the context.

    Args:
        path (Path): The path to the cwd

    Yields:
        None
    """

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)
