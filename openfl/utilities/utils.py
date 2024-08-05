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

    This function retrieves the fully qualified domain name (FQDN) of the
    system.
    If the 'FQDN' environment variable is set, its value is returned.
    Otherwise,the FQDN is determined based on the system's hostname.

    Args:
        name (str, optional): The name from which to extract the FQDN.
            Defaults to ''.

    Returns:
        str: The FQDN of the system.
    """
    fqdn = os.environ.get("FQDN", None)
    if fqdn is not None:
        return fqdn
    return getfqdn(name)


def is_fqdn(hostname: str) -> bool:
    """Check if a hostname is a fully qualified domain name.

    This function checks if a hostname is a fully qualified domain name (FQDN)
    according to the rules specified on Wikipedia.
    https://en.m.wikipedia.org/wiki/Fully_qualified_domain_name.

    Args:
        hostname (str): The hostname to check.

    Returns:
        bool: `True` if the hostname is a FQDN, `False` otherwise.
    """
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
    """Validate IP address value.

    This function checks if a string is a valid IP address.

    Args:
        address (str): The string to check.

    Returns:
        bool: `True` if the string is a valid IP address, `False` otherwise.
    """
    try:
        ipaddress.ip_address(address)
        return True
    except ValueError:
        return False


def add_log_level(level_name, level_num, method_name=None):
    """Add a new logging level to the logging module.

    This function adds a new logging level to the logging module with a
    specified name, value, and method name.

    Args:
        level_name (str): The name of the new logging level.
        level_num (int): The value of the new logging level.
        method_name (str, optional): The name of the method to use for
            the new logging level. Defaults to None.
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

    This function validates the SHA384 hash of a file against an expected hash.

    Args:
        file_path (str): The path to the file to validate.
            (absolute or relative to the current working directory) of the file
            to be opened or an integer file descriptor of the file to be
            wrapped.
        expected_hash (str): The expected SHA384 hash of the file.
        chunk_size (int, optional): The size of the chunks to read from the
            file. Defaults to 8192.

    Raises:
        SystemError: If the hash of the file does not match the expected hash.
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
    """Visualize downloading.

    This function creates a progress bar for visualizing the progress of a
    download.

    Returns:
        Callable: A function that updates the progress bar.
    """

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
    """
    Create Dynaconf settings, merge its with `overwrite_dict` and validate
    result.

    This function creates a Dynaconf settings object, merges it with an
    optional dictionary, applies an optional value transformation, and
    validates the result.

    Args:
        overwrite_dict (Optional[dict], optional): A dictionary to merge with
            the settings. Defaults to None.
        value_transform (Optional[List[Tuple[str, Callable]]], optional): A
            list of tuples, each containing a key and a function to apply to
            the value of that key. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the Dynaconf
            constructor.

    Returns:
        Dynaconf: The merged and validated settings.
    """
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

    This function adds or removes fields from tensor tags.

    Args:
        tags (Tuple[str, ...]): The tensor tags.
        add_field (str, optional): A new tensor tag field to add. Defaults to
            None.
        remove_field (str, optional): A tensor tag field to remove. Defaults
            to None.

    Returns:
        Tuple[str, ...]: The modified tensor tags.

    Raises:
        Exception: If `remove_field` is not in `tags`.
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
    """Remove a directory tree.

    This function removes a directory tree. If a file in the directory tree is
    read-only, its read-only attribute is cleared before it is removed.

    Args:
        path (str): The path to the directory tree to remove.
        ignore_errors (bool, optional): Whether to ignore errors. Defaults to
            False.

    Returns:
        str: The path to the removed directory tree.
    """

    def remove_readonly(func, path, _):
        "Clear the readonly bit and reattempt the removal"
        if os.name == "nt":
            os.chmod(path, stat.S_IWRITE)  # Windows can not remove read-only files.
        func(path)

    return shutil.rmtree(path, ignore_errors=ignore_errors, onerror=remove_readonly)
