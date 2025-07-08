import os
import sys
sys.path.insert(0, os.path.abspath('../optifik'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'optifik'
copyright = '2025, F. Boulogne et al.'
author = 'F. Boulogne et al.'
html_baseurl = "https://sciunto.github.io/optifik/"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'myst_parser',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx.ext.mathjax',
]

numpydoc_show_inherited_class_members = True
numpydoc_use_plots = False  # Plots in doctring
numpydoc_class_members_toctree = False

nbsphinx_execute = 'auto'  # ou 'auto' pour exécuter si nécessaire
nbsphinx_notebooks = ['notebooks/']  # Dossier contenant vos .ipynb

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'nature'
html_theme = "sphinx_book_theme"
html_static_path = ['_static']

html_logo = "_static/logo.png"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

