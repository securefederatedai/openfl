# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Experiment API tests module."""

import pytest

from openfl.interface.interactive_api.experiment import FLExperiment
from .test_federation import federation_object  # NOQA
# TaskInterface, DataInterface, ModelInterface,

EXPERIMENT_MAME = 'test experiment'


@pytest.fixture
def experiment_object(federation_object):  # NOQA
    """Experiment object fixture."""
    experiment_object = FLExperiment(
        federation=federation_object,
        experiment_name=EXPERIMENT_MAME)
    return experiment_object


def test_initialization(experiment_object):
    """Test experimnet object initialization."""
    assert not experiment_object.experiment_submitted
    assert experiment_object.serializer_plugin


def test_get_best_model(experiment_object):
    """Test get_best_model method."""
    model = experiment_object.get_best_model()
    assert model is None
