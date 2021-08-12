# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Workspace module."""

import os
import shutil
from pathlib import Path


class AggregatorWorkspace:
    """Aggregator workspace context manager."""

    def __init__(self, experiment_name: str, data_file_path: Path) -> None:
        """Initialize workspace context manager."""
        self.experiment_name = experiment_name
        self.data_file_path = data_file_path
        self.cwd = os.getcwd()
        self.experiment_work_dir = f'{self.cwd}/{self.experiment_name}'

    def _get_experiment_data(self):
        """Copy experiment data file and extract experiment data."""
        arch_name = f'{self.experiment_name}/{self.experiment_name}' + '.zip'
        shutil.copy(self.data_file_path, arch_name)
        shutil.unpack_archive(arch_name, self.experiment_name)
        os.remove(arch_name)

    def __enter__(self):
        """Prepare collaborator workspace."""
        if os.path.exists(self.experiment_name):
            shutil.rmtree(self.experiment_name)
        os.makedirs(self.experiment_name)

        self._get_experiment_data()

        os.chdir(self.experiment_work_dir)

    def __exit__(self, exc_type, exc_value, traceback):
        """Remove the workspace."""
        os.chdir(self.cwd)
        shutil.rmtree(self.experiment_name)
        os.remove(self.data_file_path)
