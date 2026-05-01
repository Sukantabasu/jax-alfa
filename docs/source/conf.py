# Configuration file for the Sphinx documentation builder.
import os
import sys

# Get the absolute path of the docs directory
docs_dir = os.path.abspath(os.path.dirname(__file__))
# Get the path to the project root (where your Python files are)
project_root = os.path.abspath(os.path.join(docs_dir, '../..'))
# Add it to Python's path
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'JAX-ALFA'
copyright = '2025, Sukanta Basu'
author = 'Sukanta Basu'
release = '0.0.1'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'sphinx_math_dollar',
    'sphinx.ext.autosectionlabel',
    'nbsphinx'
]

nbsphinx_execute = 'never'  # Avoid executing notebooks on build
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Add to your conf.py file
suppress_warnings = [
    'ref.duplicate',  # Suppress warnings about duplicate labels
]


def setup(app):
    from sphinx.util.logging import getLogger
    logger = getLogger('sphinx')
    
    orig_warning = logger.warning
    
    def custom_warning(message, *args, **kwargs):
        if "duplicate label" in message:
            return
        orig_warning(message, *args, **kwargs)
    
    logger.warning = custom_warning


# Add these settings
viewcode_follow_imported_members = True
add_module_names = False  # Makes function names cleaner in documentation

templates_path = ['_templates']

# HTML output configuration
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}
html_logo = "_static/logo.png"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Math settings
math_dollar_inline = True
math_dollar_display = True
