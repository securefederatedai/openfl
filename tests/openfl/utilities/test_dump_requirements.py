# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dump requirements file test module."""

from pathlib import Path

import pytest
from pip._internal.operations import freeze

from openfl.utilities.workspace import dump_requirements_file

LINES = ['-o option\n', 'package==0.0.1\n']
PREFIXES = ('-u test',)


@pytest.fixture
def requirements_file():
    """Prepare test requirements file."""
    path = Path('./test_requirements.txt').absolute()
    with open(path, 'w') as f:
        f.writelines(LINES)
    yield path
    path.unlink()


@pytest.mark.parametrize('keep_original_prefixes,prefixes,expected_lines', [
    (False, None, LINES[1:]),
    (True, None, LINES),
    (False, PREFIXES, [p + '\n' for p in PREFIXES] + [li for li in LINES if li[0] != '-']),
    (True, PREFIXES, [li for li in LINES if li[0] == '-'
                      ] + [p + '\n' for p in PREFIXES] + [li for li in LINES if li[0] != '-']),
])
def test_dump(requirements_file, monkeypatch,
              keep_original_prefixes, prefixes, expected_lines):
    """Test dump_requirements_file function."""
    def mock_pip_freeze():
        return [line.replace('\n', '') for line in LINES if line[0] != '-']
    monkeypatch.setattr(freeze, 'freeze', mock_pip_freeze)

    dump_requirements_file(path=requirements_file,
                           keep_original_prefixes=keep_original_prefixes,
                           prefixes=prefixes)

    with open(requirements_file) as f:
        read_lines = f.readlines()
    for line, expected in zip(read_lines, expected_lines):
        assert line == expected
