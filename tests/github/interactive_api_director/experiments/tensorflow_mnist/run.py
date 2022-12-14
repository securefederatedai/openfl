from pathlib import Path

from tests.github.interactive_api_director import utils


if __name__ == '__main__':
    director = utils.start_director(Path(__file__).parent / 'director')
    envoy = utils.start_envoy(Path(__file__).parent / 'envoy')
    from tests.github.interactive_api_director.experiments.tensorflow_mnist import experiment
    experiment.run()
