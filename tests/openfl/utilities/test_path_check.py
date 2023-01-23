# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Path checks tests module."""

import os
from pathlib import Path

import pytest

from openfl.utilities.path_check import is_directory_traversal


@pytest.mark.parametrize(
    'directory,expected_result', [
        ('first_level', False),
        ('first_level/second_level', False),
        (os.getcwd(), False),
        (Path(os.getcwd(), 'first_level'), False),
        (Path(os.getcwd(), 'first_level/second_level'), False),
        ('first_level/second_level/..', False),
        ('first_level/../first_level', False),
        ('..', True),
        ('../../file', True),
        ('/home/naive_hacker', True),
        ('first_level/second_level/../../..', True),
        ('..', True),
        ('../../file', True),
    ])
def test_is_directory_traversal(directory, expected_result):
    """Test that is_directory_traversal works."""
    assert is_directory_traversal(directory) is expected_result
