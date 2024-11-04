import pytest
from utils.logger import logger as log

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