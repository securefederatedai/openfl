import os
import shutil
from pathlib import Path

import yaml


if __name__ == '__main__':
    for rank in range(10):
        dst = Path(str(rank)).absolute()
        shutil.rmtree(dst)
        os.makedirs(dst)
        shutil.copy('tinyimagenet_shard_descriptor.py', dst)
        shutil.copy('base.py', dst)
        shutil.copy('requirements.txt', dst)
        shutil.copy('start_envoy_with_tls.sh', dst)
        shutil.copy('start_envoy.sh', dst)
        config = {
            'template': 'tinyimagenet_shard_descriptor.TinyImageNetShardDescriptor',
            'params': {
                'data_folder': 'tinyimagenet_data',
                'rank': rank,
                'worldsize': 10
            }
        }
        with open(dst / 'shard_config.yaml', 'w') as f:
            yaml.dump(config, f)
