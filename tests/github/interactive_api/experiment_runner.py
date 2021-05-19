# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tools for running an experiment on a local machine in multi-processing mode."""

import os
import logging
import sys
import shutil
import subprocess
import typing

from openfl.interface.interactive_api.experiment import FLExperiment
from openfl.interface.interactive_api.experiment import ModelInterface

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
)
handler.setFormatter(formatter)
root.addHandler(handler)

logger = logging.getLogger(__name__)


def prepare_collaborator_workspace(col_dir: str, arch_path: str):
    """
    Prepare a collaborator workspace.

    Args:
        col_dir (str): Collaborator directory name
        arch_path (str): Path to archive with workspace
    """
    logger.info(f'Prepare the collaborator directory: {col_dir}')
    if os.path.exists(col_dir):
        shutil.rmtree(col_dir)
    os.makedirs(col_dir)
    arch_col_path = shutil.copy(arch_path, col_dir)
    shutil.unpack_archive(arch_col_path, col_dir)
    logger.info('The collaborator directory prepared')


def run_aggregator(model_interface: ModelInterface, fl_experiment: FLExperiment):
    """
    Run the aggregator.

    Args:
        model_interface (ModelInterface): Model Interface
        fl_experiment (FLExperiment): FLExperiment
    """
    logger.info('Run the aggregator')
    fl_experiment.start_experiment(model_interface)
    logger.info('The aggregator stopped')


def run_experiment(
        col_data_paths: typing.Dict[str, str],
        model_interface: ModelInterface,
        arch_path: str,
        fl_experiment: FLExperiment
):
    """
    Run the experiment.

    Args:
        col_data_paths (typing.Dict[str, str]): Data path corresponds to 'RANK,WORLD_SIZE'
        model_interface (ModelInterface): Model interface
        arch_path (str): Path to archive with workspace
        fl_experiment (FLExperiment): FLExperiment
    """
    logger.info('Starting the experiment!')
    for col_dir in col_data_paths:
        prepare_collaborator_workspace(col_dir, arch_path)

    processes = []
    for col_name in col_data_paths:
        logger.info(f'Starting the collaborator: {col_name}')
        p = subprocess.Popen(
            f"fx collaborator start -n {col_name} -p plan/plan.yaml -d data.yaml".split(' '),
            cwd=os.path.join(os.getcwd(), col_name)
        )
        processes.append(p)

    run_aggregator(model_interface, fl_experiment)
    for p in processes:
        p.terminate()

    logger.info('The experiment completed!')
