# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Module with auxiliary CLI helper functions."""

from click import echo, style
from sys import argv
from pathlib import Path
from itertools import islice
from os import environ, stat

from yaml import load, FullLoader

FX = argv[0]

SITEPACKS = Path(__file__).parent.parent.parent
WORKSPACE = SITEPACKS / 'openfl-workspace'
TUTORIALS = SITEPACKS / 'openfl-tutorials'
PKI_DIR = Path('cert')
OPENFL_USERDIR = Path.home() / '.openfl'


def pretty(o):
    """Pretty-print the dictionary given."""
    m = max(map(len, o.keys()))

    for k, v in o.items():
        echo(style(f'{k:<{m}} : ', fg='blue') + style(f'{v}', fg='cyan'))


def tree(path):
    """Print current directory file tree."""
    echo(f'+ {path}')

    for path in sorted(path.rglob('*')):

        depth = len(path.relative_to(path).parts)
        space = '    ' * depth

        if path.is_file():
            echo(f'{space}f {path.name}')
        else:
            echo(f'{space}d {path.name}')


def print_tree(dir_path: Path, level: int = -1,
               limit_to_directories: bool = False,
               length_limit: int = 1000):
    """Given a directory Path object print a visual tree structure."""
    space = '    '
    branch = '│   '
    tee = '├── '
    last = '└── '

    echo('\nNew workspace directory structure:')

    dir_path = Path(dir_path)  # accept string coerceable to Path
    files = 0
    directories = 0

    def inner(dir_path: Path, prefix: str = '', level=-1):
        nonlocal files, directories
        if not level:
            return  # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix + extension,
                                 level=level - 1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    echo(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        echo(line)
    if next(iterator, None):
        echo(f'... length_limit, {length_limit}, reached, counted:')
    echo(f'\n{directories} directories' + (f', {files} files' if files else ''))


def copytree(src, dst, symlinks=False, ignore=None,
             ignore_dangling_symlinks=False, dirs_exist_ok=False):
    """From Python 3.8 'shutil' which include 'dirs_exist_ok' option."""
    import os
    import shutil

    with os.scandir(src) as itr:
        entries = list(itr)

    copy_function = shutil.copy2

    def _copytree():

        if ignore is not None:
            ignored_names = ignore(os.fspath(src), [x.name for x in entries])
        else:
            ignored_names = set()

        os.makedirs(dst, exist_ok=dirs_exist_ok)
        errors = []
        use_srcentry = copy_function is shutil.copy2 or \
            copy_function is shutil.copy

        for srcentry in entries:
            if srcentry.name in ignored_names:
                continue
            srcname = os.path.join(src, srcentry.name)
            dstname = os.path.join(dst, srcentry.name)
            srcobj = srcentry if use_srcentry else srcname
            try:
                is_symlink = srcentry.is_symlink()
                if is_symlink and os.name == 'nt':
                    lstat = srcentry.stat(follow_symlinks=False)
                    if lstat.st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT:
                        is_symlink = False
                if is_symlink:
                    linkto = os.readlink(srcname)
                    if symlinks:
                        os.symlink(linkto, dstname)
                        shutil.copystat(srcobj, dstname,
                                        follow_symlinks=not symlinks)
                    else:
                        if (not os.path.exists(linkto)
                                and ignore_dangling_symlinks):
                            continue
                        if srcentry.is_dir():
                            copytree(srcobj, dstname, symlinks, ignore,
                                     dirs_exist_ok=dirs_exist_ok)
                        else:
                            copy_function(srcobj, dstname)
                elif srcentry.is_dir():
                    copytree(srcobj, dstname, symlinks, ignore,
                             dirs_exist_ok=dirs_exist_ok)
                else:
                    copy_function(srcobj, dstname)
            except OSError as why:
                errors.append((srcname, dstname, str(why)))
            except Exception as err:
                errors.extend(err.args[0])
        try:
            shutil.copystat(src, dst)
        except OSError as why:
            if getattr(why, 'winerror', None) is None:
                errors.append((src, dst, str(why)))
        if errors:
            raise Exception(errors)
        return dst

    return _copytree()


def get_workspace_parameter(name):
    """Get a parameter from the workspace config file (.workspace)."""
    # Update the .workspace file to show the current workspace plan
    workspace_file = '.workspace'

    with open(workspace_file, 'r') as f:
        doc = load(f, Loader=FullLoader)

    if not doc:  # YAML is not correctly formatted
        doc = {}  # Create empty dictionary

    if name not in doc.keys() or not doc[name]:  # List doesn't exist
        return ''
    else:
        return doc[name]


def check_varenv(env="", args={}):
    """Update "args" (dictionary) with <env: env_value> if env has a defined value in the host."""
    env_val = environ.get(env)
    if env and (env_val is not None):
        args[env] = env_val

    return args


def get_fx_path(curr_path=""):
    """Return the absolute path to fx binary."""
    import re
    import os

    match = re.search("lib", curr_path)
    idx = match.end()
    path_prefix = curr_path[0:idx]
    bin_path = re.sub("lib", "bin", path_prefix)
    fx_path = os.path.join(bin_path, "fx")

    return fx_path


def remove_line_from_file(pkg, filename):
    """Remove line that contains `pkg` from the `filename` file."""
    with open(filename, "r+") as f:
        d = f.readlines()
        f.seek(0)
        for i in d:
            if pkg not in i:
                f.write(i)
        f.truncate()


def replace_line_in_file(line, line_num_to_replace, filename):
    """Replace line at `line_num_to_replace` with `line`."""
    with open(filename, "r+") as f:
        d = f.readlines()
        f.seek(0)
        for idx, i in enumerate(d):
            if idx == line_num_to_replace:
                f.write(line)
            else:
                f.write(i)
        f.truncate()
