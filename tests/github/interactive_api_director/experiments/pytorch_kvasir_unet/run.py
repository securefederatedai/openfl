import time
import sys
from subprocess import Popen, check_call
import psutil
import experiment


if __name__ == '__main__':
    bg_procs = []
    proc = Popen(
        ['fx', 'director', 'start', '--disable-tls', '-c', 'director_config.yaml'],
        cwd='director', shell=False)
    bg_procs.append(proc)
    time.sleep(3)
    if proc.pid not in psutil.pids():
        print('Error: failed to create director')
        sys.exit(1)

    check_call(
        [sys.executable, '-m', 'pip', 'install', '-r', 'sd_requirements.txt'], cwd='envoy'
    )
    proc = Popen(
        ['fx', 'envoy', 'start', '-n', 'env_one', '--disable-tls', '--envoy-config-path',
            'envoy_config.yaml', '-dh', 'localhost', '-dp', '50051'], cwd='envoy', shell=False
    )

    bg_procs.append(proc)
    time.sleep(10)
    if proc.pid not in psutil.pids():
        print('Error: failed to create envoy')
        sys.exit(1)

    experiment.start()
    for proc in bg_procs:
        proc.kill()
