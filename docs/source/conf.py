# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "abeliantensors"
copyright = "2020-2026, Markus Hauru and contributors"
author = "Markus Hauru"

master_doc = "index"
extensions = ["sphinx.ext.autodoc", "myst_parser"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"
