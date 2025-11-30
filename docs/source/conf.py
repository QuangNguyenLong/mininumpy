import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Mininumpy'
copyright = '2025, Long Quang NGUYEN'
author = 'Long Quang NGUYEN'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

templates_path = ['_templates']
exclude_patterns = []

# Show type hints in the description instead of the signature
autodoc_typehints = "description"

# Show long paths like mininumpy.core.Array
typehints_fully_qualified = True

# Always show parameter types even if napoleon-style docstrings exist
always_document_param_types = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_member_order = 'bysource'

autodoc_default_options = {
    "members": True,
    "special-members": "__init__, __str__, __add__, __mul__, __sub__, __pow__, __truediv__, __matmul__, __exp__, __sqrt__, __log__, __abs__",
}
