# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Utilities module."""

import hashlib
import ipaddress
import logging
import os
import re
import shutil
import stat
from collections.abc import Callable
from functools import partial
from socket import getfqdn
from typing import List, Optional, Tuple

from dynaconf import Dynaconf
from tqdm import tqdm


def getfqdn_env(name: str = "") -> str:
    """
    Get the system FQDN, with priority given to environment variables.

    Args:
        name: The name from which to extract the FQDN.

    Returns:
        The FQDN of the system.
    """
    fqdn = os.environ.get("FQDN", None)
    if fqdn is not None:
        return fqdn
    return getfqdn(name)


def is_fqdn(hostname: str) -> bool:
    """https://en.m.wikipedia.org/wiki/Fully_qualified_domain_name."""
    if not 1 < len(hostname) < 253:
        return False

    # Remove trailing dot
    hostname.rstrip(".")

    #  Split hostname into list of DNS labels
    labels = hostname.split(".")

    #  Define pattern of DNS label
    #  Can begin and end with a number or letter only
    #  Can contain hyphens, a-z, A-Z, 0-9
    #  1 - 63 chars allowed
    fqdn = re.compile(r"^[a-z0-9]([a-z-0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)  # noqa FS003

    # Check that all labels match that pattern.
    return all(fqdn.match(label) for label in labels)


def is_api_adress(address: str) -> bool:
    """Validate ip address value."""
    try:
        ipaddress.ip_address(address)
        return True
    except ValueError:
        return False


def add_log_level(level_name, level_num, method_name=None):
    """
    Add a new logging level to the logging module.

    Args:
        level_name: name of log level.
        level_num: log level value.
        method_name: log method wich will use new log level (default = level_name.lower())

    """
    if not method_name:
        method_name = level_name.lower()

    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)


def validate_file_hash(file_path, expected_hash, chunk_size=8192):
    """Validate SHA384 hash for file specified.

    Args:
        file_path(path-like): path-like object giving the pathname
            (absolute or relative to the current working directory)
            of the file to be opened or an integer file descriptor of the file to be wrapped.
        expected_hash(str): hash string to compare with.
        hasher(_Hash): hash algorithm. Default value: `hashlib.sha384()`
        chunk_size(int): Buffer size for file reading.
    """
    h = hashlib.sha384()
    with open(file_path, "rb") as file:
        # Reading is buffered, so we can read smaller chunks.
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    if h.hexdigest() != expected_hash:
        raise SystemError("ZIP File hash doesn't match expected file hash.")


def tqdm_report_hook():
    """Visualize downloading."""

    def report_hook(pbar, count, block_size, total_size):
        """Update progressbar."""
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    pbar = tqdm(total=None)
    return partial(report_hook, pbar)


def merge_configs(
    overwrite_dict: Optional[dict] = None,
    value_transform: Optional[List[Tuple[str, Callable]]] = None,
    **kwargs,
) -> Dynaconf:
    """Create Dynaconf settings, merge its with `overwrite_dict` and validate result."""
    settings = Dynaconf(**kwargs, YAML_LOADER="safe_load")
    if overwrite_dict:
        for key, value in overwrite_dict.items():
            if value is not None or settings.get(key) is None:
                settings.set(key, value, merge=True)
    if value_transform:
        for key, operation in value_transform:
            value = settings.get(key)
            settings.set(key, operation(value))
    settings.validators.validate()
    return settings


def change_tags(tags, *, add_field=None, remove_field=None) -> Tuple[str, ...]:
    """Change tensor tags to add or remove fields.

    Args:
        tags(tuple): tensor tags.
        add_field(str): add a new tensor tag field.
        remove_field(str): remove a tensor tag field.
    """
    tags = list(set(tags))

    if add_field is not None and add_field not in tags:
        tags.append(add_field)
    if remove_field is not None:
        if remove_field in tags:
            tags.remove(remove_field)
        else:
            raise Exception(f"{remove_field} not in tags {tuple(tags)}")

    tags = tuple(sorted(tags))
    return tags


def rmtree(path, ignore_errors=False):
    def remove_readonly(func, path, _):
        "Clear the readonly bit and reattempt the removal"
        if os.name == "nt":
            os.chmod(path, stat.S_IWRITE)  # Windows can not remove read-only files.
        func(path)

    return shutil.rmtree(path, ignore_errors=ignore_errors, onerror=remove_readonly)
