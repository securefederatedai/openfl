import pytest
from unittest import mock

from .test_federation import federation_object

from openfl.interface.interactive_api.experiment import FLExperiment
# TaskInterface, DataInterface, ModelInterface,

EXPERIMENT_MAME = 'test experiment'


@pytest.fixture
def experiment_object(federation_object):
    experiment_object = FLExperiment(
        federation=federation_object,
        experiment_name=EXPERIMENT_MAME)
    return experiment_object


def test_initialization(experiment_object):
    assert not experiment_object.experiment_accepted
    assert experiment_object.serializer_plugin

def test_get_best_model(experiment_object):
    with pytest.raises(Exception):
        experiment_object.get_best_model()
