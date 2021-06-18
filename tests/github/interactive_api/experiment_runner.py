import logging
import os
import shutil
import subprocess

from openfl.utilities.logs import setup_loggers

setup_loggers(logging.INFO)
logger = logging.getLogger(__name__)


def prepare_collaborator_workspace(col_dir, arch_path):
    logger.info(f'Prepare collaborator directory: {col_dir}')
    if os.path.exists(col_dir):
        shutil.rmtree(col_dir)
    os.makedirs(col_dir)
    arch_col_path = shutil.copy(arch_path, col_dir)
    shutil.unpack_archive(arch_col_path, col_dir)
    logger.info('Collaborator directory prepared')


def run_aggregator(model_interface, fl_experiment):
    logger.info('run_aggregator')
    fl_experiment.start_experiment(model_interface)
    logger.info('Aggregator stopped')


def run_experiment(col_data_paths, model_interface, arch_path, fl_experiment):
    logger.info('Starting the experiment!')
    for col_dir in col_data_paths:
        prepare_collaborator_workspace(col_dir, arch_path)

    processes = []
    for col_name in col_data_paths:
        logger.info(f'Starting collaborator: {col_name}')
        p = subprocess.Popen(
            f'fx collaborator start -n {col_name} -p plan/plan.yaml -d data.yaml'.split(' '),
            cwd=os.path.join(os.getcwd(), col_name)
        )
        processes.append(p)

    run_aggregator(model_interface, fl_experiment)
    for p in processes:
        p.terminate()

    logger.info('The experiment completed!')
