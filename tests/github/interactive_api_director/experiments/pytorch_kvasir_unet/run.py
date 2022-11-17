import time
import sys
from subprocess import Popen, check_call
import psutil
import importlib
from pathlib import Path

if __name__ == '__main__':
    director_dir = Path(__file__).parent / 'director'
    bg_procs = []
    proc = Popen(
        ['fx', 'director', 'start', '--disable-tls', '-c', 'director_config.yaml'],
        cwd=director_dir, shell=False)
    bg_procs.append(proc)
    time.sleep(3)
    if proc.pid not in psutil.pids():
        print('Error: failed to create director')
        sys.exit(1)

    envoy_dir = Path(__file__).parent / 'envoy'
    check_call(
        [sys.executable, '-m', 'pip', 'install', '-r', 'sd_requirements.txt'], cwd=envoy_dir
    )
    proc = Popen(
        ['fx', 'envoy', 'start', '-n', 'env_one', '--disable-tls', '--envoy-config-path',
            'envoy_config.yaml', '-dh', 'localhost', '-dp', '50051'], cwd=envoy_dir, shell=False
    )

    bg_procs.append(proc)
    time.sleep(10)
    if proc.pid not in psutil.pids():
        print('Error: failed to create envoy')
        sys.exit(1)
    experiment = importlib.import_module(
        'tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.experiment'
    )
    experiment.run()
    for proc in bg_procs:
        proc.kill()
