# -- Path setup --------------------------------------------------------------

import os
import re
import subprocess
import sys
import warnings

import scprinter as scp

# -- Project information -----------------------------------------------------

project = "scprinter"
author = "Ruochi Zhang, Yan Hu"

# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
version = re.sub(r"(\d+\.\d+)\.\d+(.*)", r"\1\2", scp.__version__)
version = re.sub(r"(\.dev\d+).*?$", r"\1", version)
# The full version, including alpha/beta/rc tags.
release = scp.__version__
print("%s %s" % (version, release))

# -- General configuration ---------------------------------------------------

suppress_warnings = ["ref.citation"]
default_role = "code"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
    "sphinx_plotly_directive",
    "sphinxcontrib.video",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

myst_enable_extensions = [
    "amsmath",
    # "colon_fence",
    # "deflist",
    "dollarmath",
    # "fieldlist",
    # "html_admonition",
    # "html_image",
    # "linkify",
    # "replacements",
    # "smartquotes",
    # "strikethrough",
    # "substitution",
    # "tasklist",
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
# autodoc_default_flags = ['members']
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False

intersphinx_mapping = {
    "cycler": ("https://matplotlib.org/cycler/", None),
    "h5py": ("http://docs.h5py.org/en/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "networkx": (
        "https://networkx.github.io/documentation/networkx-1.10/",
        None,
    ),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pytest": ("https://docs.pytest.org/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

smv_branch_whitelist = r"main"  # Include all branches

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_show_sphinx = False
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
if ".dev" in version:
    switcher_version = "dev"
else:
    switcher_version = f"{version}"

html_theme_options = {
    "github_url": "https://github.com/buenrostrolab/scPrinter",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # "switcher": {
    #     "version_match": switcher_version,
    #     "json_url": "https://raw.githubusercontent.com/broadinstitute/PRINT/main/docs/_static/versions.json",
    # },
}


# commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
commit = "main"
code_url = f"https://github.com/buenrostrolab/scPrinter/blob/{commit}"


# wrapping out really long outputs in notebooks
nbsphinx_prolog = r"""
.. raw:: html

    <style>
    .output-area-wrapper {
        border: 1px solid #ccc;
        padding: 8px;
        margin-bottom: 1em;
        background-color: #f9f9f9;
        position: relative;
    }
    .toggle-button-wrapper {
        text-align: center;
        margin-bottom: 10px;
    }
    </style>

    <script>
    document.addEventListener("DOMContentLoaded", function () {
        const all = Array.from(document.querySelectorAll('.nboutput'));
        let group = [];

        const flushGroup = () => {
            if (group.length === 0) return;

            // Combine all outputs inside group
            let totalText = '';
            group.forEach(nb => {
                const outs = nb.querySelectorAll('.output_area');
                outs.forEach(out => totalText += out.innerText);
            });

            if (totalText.length > 300) {
                const wrapper = document.createElement('div');
                wrapper.className = 'output-area-wrapper';

                // Create and insert toggle button inside wrapper
                const buttonDiv = document.createElement('div');
                buttonDiv.className = 'toggle-button-wrapper';
                const button = document.createElement('button');
                button.innerText = 'Show/Hide Output';
                button.onclick = () => {
                    outputsDiv.style.display = (outputsDiv.style.display === 'none') ? 'block' : 'none';
                };
                buttonDiv.appendChild(button);
                wrapper.appendChild(buttonDiv);

                // Create div for actual outputs
                const outputsDiv = document.createElement('div');
                group.forEach(nb => {
                    const outs = Array.from(nb.querySelectorAll('.output_area'));
                    outs.forEach(out => outputsDiv.appendChild(out));
                });
                outputsDiv.style.display = 'none';
                wrapper.appendChild(outputsDiv);

                // Insert wrapper before first .nboutput
                const first = group[0];
                first.parentNode.insertBefore(wrapper, first);
                group.forEach(nb => nb.remove());
            }

            group = [];
        };

        for (let i = 0; i < all.length; i++) {
            const el = all[i];
            if (i === 0 || el.previousElementSibling?.classList.contains('nboutput')) {
                group.push(el);
            } else {
                flushGroup();
                group.push(el);
            }
        }
        flushGroup();
    });
    </script>
"""


# based on numpy doc/source/conf.py
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    import inspect

    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            with warnings.catch_warnings():
                # Accessing deprecated objects will generate noisy warnings
                warnings.simplefilter("ignore", FutureWarning)
                obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(scp.__file__))

    return f"{code_url}/scprinter/{fn}{linespec}"
