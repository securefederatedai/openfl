# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shared test configurations for transport tests."""

import pytest
import logging
from pathlib import Path


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    logging.basicConfig(level=logging.DEBUG)
    yield


@pytest.fixture(autouse=True)
def mock_environment(monkeypatch):
    """Mock environment variables and system settings."""
    monkeypatch.setenv('PYTHONPATH', '')  # Clear PYTHONPATH to avoid interference
    yield


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / 'data'


@pytest.fixture(autouse=True)
def setup_test_data(test_data_dir):
    """Set up test data directory."""
    test_data_dir.mkdir(exist_ok=True)
    yield
    # Cleanup after tests if needed
    if test_data_dir.exists():
        for file in test_data_dir.glob('*'):
            if file.is_file():
                file.unlink()
