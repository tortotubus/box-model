import math
import os
from os.path import relpath, dirname
import re
import sys
import warnings
from datetime import date
from docutils import nodes
from docutils.parsers.rst import Directive

from numpydoc.docscrape_sphinx import SphinxDocString
from sphinx.util import inspect

import boxmodel

#old_isdesc = inspect.isdescriptor
#inspect.isdescriptor = (lambda obj: old_isdesc(obj)
#                        and not isinstance(obj, ua._Function))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Box Model'
copyright = '2023, Conor Olive'
author = 'Conor Olive'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import numpydoc.docscrape as np_docscrape  # noqa:E402

#sys.path.insert(0, os.path.abspath(os.path.dirname('../../')))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_design',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'BoxModel'
default_role = "autolink"
add_function_parenthesis = False
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------

autodoc_default_options = {
    'inherited-members': None,
    'special-members': '__init__',
}
autodoc_typehints = 'none'

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
