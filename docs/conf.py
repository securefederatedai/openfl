# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Docs configuration module."""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import inspect
import operator
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('../'))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# import sphinx_rtd_theme # NOQA:E800
# import sphinxcontrib.napoleon # NOQA:E800

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx_remove_toctrees',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxext.rediraffe',
    'sphinxcontrib.mermaid',
    'sphinx-prompt',
    'recommonmark',
    # 'sphinx.ext.ifconfig',
    # 'sphinx.ext.autosectionlabel',
    # 'sphinx_substitution_extensions',
]
autodoc_default_options = {
    'imported-members': True,
}
autosummary_generate = True  # Turn on sphinx.ext.autosummary

source_suffix = ['.rst', '.md']

# -- Project information -----------------------------------------------------
# This will replace the |variables| within the rST documents automatically
project = 'OpenFL'
copyright = f'{datetime.now().year}, The OpenFL Team'
author = 'The OpenFL Team'
version = ''
release = ''
main_doc = 'index'

napoleon_google_docstring = True

# Config the returns section to behave like the Args section
napoleon_custom_sections = [('Returns', 'params_style')]

# This code extends Sphinx's GoogleDocstring class to support 'Keys',
# 'Attributes', and 'Class Attributes' sections in docstrings. Allows for more
# detailed and structured documentation of Python classes and their attributes.
from sphinx.ext.napoleon.docstring import GoogleDocstring # NOQA

# Define new sections and their corresponding parse methods
new_sections = {
    'keys': 'Keys',
    'attributes': 'Attributes',
    'class attributes': 'Class Attributes'
}

# Add new sections to GoogleDocstring class
for section, title in new_sections.items():
    setattr(GoogleDocstring, f'_parse_{section}_section',
            lambda self, section: self._format_fields(title, self._consume_fields()))


# Patch the parse method to include new sections
def patched_parse(self):
    for section in new_sections:
        self._sections[section] = getattr(self, f'_parse_{section}_section')
    self._unpatched_parse()


# Apply the patch
GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', 'README.md', 'structurizer_dsl/README.md',
                    '.DS_Store', 'tutorials/*', 'graveyard/*', '_templates']

# add temporary unused files
exclude_patterns.extend(['install.singularity.rst',
                         'overview.what_is_intel_federated_learning.rst',
                         'overview.how_can_intel_protect_federated_learning.rst',
                         'source/workflow/running_the_federation.singularity.rst'])

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_logo = '_static/openfl_logo.png'
html_favicon = '_static/favicon.png'
html_theme = 'sphinx_book_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/securefederatedai/openfl',
    'use_repository_button': True,     # add a "link to repository" button
    'navigation_with_keys': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# html_style = 'css/Intel_One_Mono_Font_Theme.css'
# Customize code links via sphinx.ext.linkcode

def linkcode_resolve(domain, info):
  import openfl

  if domain != 'py':
    return None
  if not info['module']:
    return None
  if not info['fullname']:
    return None
  if info['module'].split(".")[0] != 'openfl':
     return None
  try:
    mod = sys.modules.get(info['module'])
    obj = operator.attrgetter(info['fullname'])(mod)
    if isinstance(obj, property):
        obj = obj.fget
    while hasattr(obj, '__wrapped__'):  # decorated functions
        obj = obj.__wrapped__
    filename = inspect.getsourcefile(obj)
    source, linenum = inspect.getsourcelines(obj)
  except:
    return None
  filename = os.path.relpath(filename, start=os.path.dirname(openfl.__file__))
  lines = f"#L{linenum}-L{linenum + len(source)}" if linenum else ""
  return f"https://github.com/securefederatedai/openfl/blob/develop/openfl/{filename}{lines}"