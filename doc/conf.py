import nabu

from nabu._version import __version__

project = "NABU"
copyright = "2024, The NABU Authors"
author = "Jack Y. Araz and others"
master_doc = "index"

release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinxarg.ext",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
