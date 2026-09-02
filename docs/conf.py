"""Sphinx configuration for the Prothon documentation."""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Prothon"
author = "Adekunle Aina"
copyright = "2026, AAI Research Lab"

try:
    from prothon import __version__ as release
except ImportError:  # docs can be built without the package installed
    release = "2.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# mdtraj compiles extension modules; Read the Docs installs it, but mocking
# keeps a docs build from failing on a wheel that is missing for one platform.
autodoc_mock_imports = ["mdtraj"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Render an "Attributes" section as :ivar: fields rather than as separate
# attribute directives. Without this, a dataclass attribute described in the
# class docstring and also picked up by autodoc is documented twice, and
# Sphinx reports every one of them as a duplicate.
napoleon_use_ivar = True

# An unreachable inventory is a network problem rather than a documentation
# problem. Sphinx does not give that warning a suppressible type, so the CI job
# filters it out by text instead and fails on everything else.
intersphinx_timeout = 10

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# An unreachable inventory is a condition of somebody else's server, not a
# defect in this documentation. The build is otherwise complete; the only loss
# is that `numpy.ndarray` renders unlinked. Warnings are errors here, so a
# timeout at docs.scipy.org failed the first Read the Docs build after
# warnings-as-errors was turned on.
#
# `suppress_warnings` cannot help. Intersphinx emits this through a bare
# `LOGGER.warning` with no type or subtype, so there is no name to suppress.
# A filter on the message is the remaining option, and it is narrow: this one
# message, from this one logger.
#
# Every other warning stays fatal, which is the point of turning them into
# errors. A page missing from a toctree, a reference that does not resolve, a
# docstring Sphinx cannot parse -- those are ours and they should stop the
# build.
intersphinx_timeout = 5


class _UnreachableInventory(logging.Filter):
    """Drop the "failed to reach any of the inventories" warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "failed to reach any of the inventories" not in record.getMessage()


def setup(app):
    """Attach the filter before intersphinx runs.

    The logger is named `sphinx.sphinx.ext.intersphinx` -- Sphinx prefixes its
    own name to the module's, and the module is `_load`, so neither the import
    path nor the obvious guess is right. Read it from the object rather than
    writing it out, so this keeps working if the module moves again.
    """
    import sphinx.ext.intersphinx._load as loader

    loader.LOGGER.logger.addFilter(_UnreachableInventory())
    return {"parallel_read_safe": True, "parallel_write_safe": True}


source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Prothon"

html_theme_options = {
    "style_external_links": True,
    "collapse_navigation": False,
    "navigation_depth": 2,
}

# Puts an "Edit on GitHub" link in the header of every page, so the source of
# any claim in these docs is one click away.
html_context = {
    "display_github": True,
    "github_user": "aai-research-lab",
    "github_repo": "Prothon",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
