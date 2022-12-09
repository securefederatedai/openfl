import time
import sys
from subprocess import Popen, check_call
import psutil
from pathlib import Path


def start_director():
    director_dir = Path(__file__).parent / 'director'
    director = Popen('fx director start '
                     '--disable-tls '
                     '-c director_config.yaml',
                     cwd=director_dir, shell=True)
    time.sleep(3)
    if director.pid not in psutil.pids():
        print('Error: failed to create director')
        sys.exit(1)
    return director


def start_envoy():
    envoy_dir = Path(__file__).parent / 'envoy'
    check_call(
        [sys.executable, '-m', 'pip', 'install', '-r', 'sd_requirements.txt'], cwd=envoy_dir
    )
    envoy = Popen('fx envoy start '
                  '-n env_one '
                  '--disable-tls '
                  '--envoy-config-path envoy_config.yaml '
                  '-dh localhost '
                  '-dp 50051',
                  cwd=envoy_dir, shell=True)
    time.sleep(10)
    if envoy.pid not in psutil.pids():
        print('Error: failed to create envoy')
        sys.exit(1)
    return envoy


if __name__ == '__main__':
    director = start_director()
    envoy = start_envoy()
    from tests.github.interactive_api_director.experiments.pytorch_kvasir_unet import experiment
    experiment.run()
