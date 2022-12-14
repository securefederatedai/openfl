from pathlib import Path

from tests.github.interactive_api_director import utils


if __name__ == '__main__':
    director = utils.start_director(Path(__file__).parent / 'director', 'director_config.yaml')
    envoy = utils.start_envoy(Path(__file__).parent / 'envoy', 'envoy_config.yaml')
    from tests.github.interactive_api_director.experiments.pytorch_kvasir_unet import experiment
    experiment.run()
