# Copyright 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.openfl_e2e.utils.logger import logger as log


# This file contains sample test functions to be run by pytest

# Function to be tested
def add(a, b):
    return a + b


# Test function
def test_add():
    log.info("Running test_add")
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    log.info("test_add passed")
