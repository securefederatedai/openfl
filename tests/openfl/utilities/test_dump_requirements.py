# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dump requirements file test module."""

from pathlib import Path

import pytest
from pip._internal.operations import freeze

from openfl.utilities.workspace import dump_requirements_file

LINES = ['-o option\n', 'package==0.0.1\n']
PREFIXES = ('-u test',)
PREFIXES_OVERLAP = ('-u test', '-o option')


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
    (True, PREFIXES_OVERLAP, [p + '\n' for p in PREFIXES_OVERLAP
                              ] + [li for li in LINES if li[0] != '-']),
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

    read_options = []
    for li in read_lines:
        if li[0] == '-':
            read_options.append(li)
        else:
            break
    read_packages = read_lines[len(read_options):]

    expected_options = []
    for li in expected_lines:
        if li[0] == '-':
            expected_options.append(li)
        else:
            break
    expected_packages = expected_lines[len(expected_options):]

    assert len(read_options) == len(expected_options)
    assert len(read_packages) == len(expected_packages)
    assert set(read_options) == set(expected_options)
    assert set(read_packages) == set(expected_packages)
