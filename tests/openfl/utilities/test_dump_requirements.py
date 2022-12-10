# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dump requirements file test module."""

from pathlib import Path

import pytest
from pip._internal.operations import freeze

from openfl.utilities.workspace import dump_requirements_file

ORIGINAL_PREFIXE_LINES = ['-o option\n', ]
ORIGINAL_PACKAGE_LINES = ['package==0.0.1\n', ]
LINES = ORIGINAL_PREFIXE_LINES + ORIGINAL_PACKAGE_LINES

PREFIXES = ('-u test',)
NEW_PREFIX_LINES = [p + '\n' for p in PREFIXES]
PREFIXES_OVERLAP = ('-u test', '-o option')
NEW_PREFIXES_OVERLAP = [p + '\n' for p in PREFIXES_OVERLAP]


@pytest.fixture
def requirements_file():
    """Prepare test requirements file."""
    path = Path('./test_requirements.txt').absolute()
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(LINES)
    yield path
    path.unlink()


@pytest.mark.parametrize('keep_original_prefixes,prefixes,expected_lines', [
    (False, None, LINES[1:]),
    (True, None, LINES),
    (False, PREFIXES, NEW_PREFIX_LINES + ORIGINAL_PACKAGE_LINES),
    (True, PREFIXES, NEW_PREFIX_LINES + LINES),
    (True, PREFIXES_OVERLAP, NEW_PREFIXES_OVERLAP + ORIGINAL_PACKAGE_LINES),
])
def test_dump(requirements_file, monkeypatch,
              keep_original_prefixes, prefixes, expected_lines):
    """Test dump_requirements_file function."""
    def mock_pip_freeze():
        return [line.replace('\n', '') for line in ORIGINAL_PACKAGE_LINES]
    monkeypatch.setattr(freeze, 'freeze', mock_pip_freeze)

    dump_requirements_file(path=requirements_file,
                           keep_original_prefixes=keep_original_prefixes,
                           prefixes=prefixes)

    with open(requirements_file, encoding='utf-8') as f:
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


@pytest.mark.parametrize('touch_file,keep_original_prefixes,prefixes,expected_lines', [
    (False, False, None, LINES[1:]),
    (True, False, None, LINES[1:]),
    (False, True, None, LINES[1:]),
    (True, True, None, LINES[1:]),
    (False, False, PREFIXES, NEW_PREFIX_LINES + ORIGINAL_PACKAGE_LINES),
    (True, False, PREFIXES, NEW_PREFIX_LINES + ORIGINAL_PACKAGE_LINES),
    (False, True, PREFIXES, NEW_PREFIX_LINES + ORIGINAL_PACKAGE_LINES),
    (True, True, PREFIXES, NEW_PREFIX_LINES + ORIGINAL_PACKAGE_LINES),
    (False, True, PREFIXES_OVERLAP, NEW_PREFIXES_OVERLAP + ORIGINAL_PACKAGE_LINES),
    (True, True, PREFIXES_OVERLAP, NEW_PREFIXES_OVERLAP + ORIGINAL_PACKAGE_LINES),
])
def test_dump_empty_original_list(
        monkeypatch, touch_file,
        keep_original_prefixes, prefixes, expected_lines):
    """Test dump_requirements_file function with no file to start."""
    def mock_pip_freeze():
        return [line.replace('\n', '') for line in ORIGINAL_PACKAGE_LINES]
    monkeypatch.setattr(freeze, 'freeze', mock_pip_freeze)

    requirements_file = Path('./test_requirements_2.txt').absolute()
    if touch_file:
        requirements_file.touch()
    try:
        dump_requirements_file(path=requirements_file,
                               keep_original_prefixes=keep_original_prefixes,
                               prefixes=prefixes)

        with open(requirements_file, encoding='utf-8') as f:
            read_lines = f.readlines()
    finally:
        requirements_file.unlink(missing_ok=True)

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
