import subprocess
import time
import sys


def start_director(cwd):
    director = subprocess.Popen(
        'fx director start '
        '--disable-tls '
        '-c director_config.yaml',
        cwd=cwd, shell=True
    )
    time.sleep(3)
    if director.poll() is not None:
        print('Error: failed to create director')
        sys.exit(1)
    return director


def start_envoy(cwd):
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-r', 'sd_requirements.txt'], cwd=cwd
    )
    envoy = subprocess.Popen(
        'fx envoy start '
        '-n env_one '
        '--disable-tls '
        '--envoy-config-path envoy_config.yaml '
        '-dh localhost '
        '-dp 50051',
        cwd=cwd, shell=True
    )
    time.sleep(10)
    if envoy.poll() is not None:
        print('Error: failed to create envoy')
        sys.exit(1)
    return envoy
