# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataloading utility functions."""

import os
import tempfile
import zipfile
from unittest import mock

import pytest

from openfl.federated import Plan
from openfl.utilities.dataloading import (
    _get_collaborator_dataloader,
    _get_minimal_dataloader,
    get_dataloader,
)
from openfl.utilities.mocks import MockDataLoader


class TestDataloading:
    """Test cases for dataloading functions."""

    @pytest.fixture
    def mock_plan(self):
        """Create a mock plan with basic configuration."""
        plan = mock.MagicMock(spec=Plan)
        plan.cols_data_paths = {"one": "data/one", "two": "data/two"}
        plan.config = {
            "data_loader": {
                "settings": {
                    "batch_size": 32,
                    "input_shape": [1, 28, 28],
                }
            }
        }
        # Mock the get_data_loader method to return a mock dataloader
        plan.get_data_loader.return_value = mock.MagicMock()
        return plan

    @pytest.fixture
    def seed_data_zip(self):
        """Create a temporary zip file to use as seed data."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            # Create a simple zip file with a test file inside
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                zipf.writestr('test.txt', 'Test data for seed')

            yield tmp_file.name

            # Clean up after the test
            os.unlink(tmp_file.name)

    def test_get_dataloader_prefer_minimal_true(self, mock_plan):
        """Test that get_dataloader uses _get_minimal_dataloader when prefer_minimal=True."""
        with mock.patch('openfl.utilities.dataloading._get_minimal_dataloader') as mock_minimal:
            get_dataloader(mock_plan, prefer_minimal=True)
            mock_minimal.assert_called_once_with(mock_plan, None)

    def test_get_dataloader_prefer_minimal_false(self, mock_plan):
        """Test that get_dataloader uses _get_collaborator_dataloader when prefer_minimal=False."""
        with mock.patch('openfl.utilities.dataloading._get_collaborator_dataloader') as mock_collab:
            get_dataloader(mock_plan, prefer_minimal=False)
            mock_collab.assert_called_once_with(mock_plan, 0)

    def test_get_dataloader_with_input_shape_and_index(self, mock_plan):
        """Test that get_dataloader passes input_shape and collaborator_index correctly."""
        input_shape = [3, 224, 224]
        collaborator_index = 1

        with mock.patch('openfl.utilities.dataloading._get_minimal_dataloader') as mock_minimal:
            get_dataloader(
                mock_plan,
                prefer_minimal=True,
                input_shape=input_shape,
                collaborator_index=collaborator_index,
            )
            mock_minimal.assert_called_once_with(mock_plan, input_shape)

        with mock.patch('openfl.utilities.dataloading._get_collaborator_dataloader') as mock_collab:
            get_dataloader(
                mock_plan,
                prefer_minimal=False,
                input_shape=input_shape,
                collaborator_index=collaborator_index,
            )
            mock_collab.assert_called_once_with(mock_plan, collaborator_index)

    def test_get_minimal_dataloader_with_input_shape(self, mock_plan):
        """Test _get_minimal_dataloader creates a MockDataLoader when input_shape is provided."""
        input_shape = [3, 224, 224]
        dataloader = _get_minimal_dataloader(mock_plan, input_shape)

        assert isinstance(dataloader, MockDataLoader)
        # Check that the input shape is passed correctly
        assert dataloader.input_shape == input_shape
        # Check that settings from the plan are applied
        assert dataloader.batch_size == 32

    def test_get_minimal_dataloader_uses_plan_input_shape(self, mock_plan):
        """Test _get_minimal_dataloader uses input_shape from plan when not explicitly provided."""
        dataloader = _get_minimal_dataloader(mock_plan)

        assert isinstance(dataloader, MockDataLoader)
        # Check that the plan's input shape is used
        assert dataloader.input_shape == [1, 28, 28]

    def test_get_minimal_dataloader_missing_input_shape(self, mock_plan):
        """Test _get_minimal_dataloader raises exception when no input_shape is available."""
        # Remove input_shape from plan config
        mock_plan.config["data_loader"]["settings"].pop("input_shape")

        # Should raise ValueError when no input_shape is provided
        with pytest.raises(ValueError) as exc_info:
            _get_minimal_dataloader(mock_plan, None)

        # Verify the error message mentions input_shape requirement
        assert "input_shape is required" in str(exc_info.value)
        assert "fx plan initialize" in str(exc_info.value)

    def test_get_collaborator_dataloader_invalid_index(self, mock_plan):
        """Test _get_collaborator_dataloader raises exception for invalid collaborator index."""
        with pytest.raises(Exception) as exc_info:
            _get_collaborator_dataloader(mock_plan, collaborator_index=5)

        assert "Unable to construct dataloader for index=5" in str(exc_info.value)

    def test_get_collaborator_dataloader_seed_data(self, mock_plan, seed_data_zip, tmp_path):
        """Test _get_collaborator_dataloader extracts seed data when provided."""
        # Set up a data path in a temporary directory
        data_path = tmp_path / "data" / "one"
        mock_plan.cols_data_paths = {"one": str(data_path)}

        # Add seed_data to the plan config
        mock_plan.config["data_loader"]["settings"]["seed_data"] = seed_data_zip

        # Call the function
        _get_collaborator_dataloader(mock_plan)

        # Verify that the seed data was extracted
        assert os.path.exists(data_path)
        assert os.path.isfile(os.path.join(data_path, "test.txt"))

        # Verify get_data_loader was called with the right name
        mock_plan.get_data_loader.assert_called_once_with("one")

    def test_get_collaborator_dataloader_seed_data_not_found(self, mock_plan, tmp_path):
        """Test _get_collaborator_dataloader handles missing seed data file gracefully."""
        # Set up a data path in a temporary directory
        data_path = tmp_path / "data" / "one"
        os.makedirs(data_path, exist_ok=True)
        mock_plan.cols_data_paths = {"one": str(data_path)}

        # Add non-existent seed_data to the plan config
        mock_plan.config["data_loader"]["settings"]["seed_data"] = "/path/does/not/exist.zip"

        # Mock logger to capture warning
        with mock.patch('logging.getLogger') as mock_get_logger:
            mock_logger = mock.MagicMock()
            mock_get_logger.return_value = mock_logger

            # Call the function
            _get_collaborator_dataloader(mock_plan)

            # Verify that a warning was logged
            mock_logger.warning.assert_called_once()
            assert "but file not found" in mock_logger.warning.call_args[0][0]

            # Verify get_data_loader was still called with the right name
            mock_plan.get_data_loader.assert_called_once_with("one")

    def test_get_collaborator_dataloader_seed_data_existing_directory(
        self, mock_plan, seed_data_zip, tmp_path
    ):
        """Test _get_collaborator_dataloader overwrites existing directory
        when seed data is provided."""
        # Create a data path that already exists
        data_path = tmp_path / "data" / "one"
        os.makedirs(data_path, exist_ok=True)
        # Create a dummy file in the directory to verify it gets overwritten
        with open(os.path.join(data_path, "old_file.txt"), 'w') as f:
            f.write("This should be overwritten")

        mock_plan.cols_data_paths = {"one": str(data_path)}

        # Add seed_data to the plan config
        mock_plan.config["data_loader"]["settings"]["seed_data"] = seed_data_zip

        # Call the function
        _get_collaborator_dataloader(mock_plan)

        # Verify that the seed data was extracted
        assert os.path.exists(data_path)
        assert os.path.isfile(os.path.join(data_path, "test.txt"))

        # Verify get_data_loader was called with the right name
        mock_plan.get_data_loader.assert_called_once_with("one")

    def test_get_collaborator_dataloader_no_seed_data(self, mock_plan):
        """Test _get_collaborator_dataloader works properly without seed data."""
        # Call the function without seed data in config
        data_loader = _get_collaborator_dataloader(mock_plan)

        # Verify get_data_loader was called with the right collaborator name
        mock_plan.get_data_loader.assert_called_once_with("one")
        # Verify the returned dataloader is what get_data_loader returned
        assert data_loader == mock_plan.get_data_loader.return_value

    def test_get_collaborator_dataloader_empty_data_yaml(self, mock_plan):
        """Test _get_collaborator_dataloader with empty data.yaml file."""
        # Empty cols_data_paths to simulate empty data.yaml
        mock_plan.cols_data_paths = {}

        with pytest.raises(Exception) as exc_info:
            _get_collaborator_dataloader(mock_plan)

        assert "when the plan has 0 total collaborators" in str(exc_info.value)

    def test_integration_with_minimal_dataloader(self, mock_plan):
        """Test integration between get_dataloader and _get_minimal_dataloader."""
        # Test that the full pipeline works with prefer_minimal=True
        dataloader = get_dataloader(mock_plan, prefer_minimal=True, input_shape=[3, 224, 224])

        assert isinstance(dataloader, MockDataLoader)
        assert dataloader.input_shape == [3, 224, 224]

    def test_integration_with_collaborator_dataloader(self, mock_plan):
        """Test integration between get_dataloader and _get_collaborator_dataloader."""
        # Mock os.path.exists to bypass the data directory check
        with mock.patch('os.path.isdir', return_value=True):
            # Test that the full pipeline works with prefer_minimal=False
            dataloader = get_dataloader(mock_plan, prefer_minimal=False, collaborator_index=0)

            # Verify that the right collaborator name was used
            mock_plan.get_data_loader.assert_called_once_with("one")
            # Verify the returned dataloader is what get_data_loader returned
            assert dataloader == mock_plan.get_data_loader.return_value
