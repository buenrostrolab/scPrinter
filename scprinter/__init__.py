import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # For Python <3.8 compatibility
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("scprinter")  # Get the version from the installed package
except PackageNotFoundError:
    __version__ = "0.0.0"  # Default when running directly from source without install

from . import chromvar, datasets, dorc, genome, motifs, peak
from . import plotting as pl
from . import preprocessing as pp
from . import seq
from . import tools as tl
from . import utils
from .io import load_printer, scPrinter
from .seq import Models, interpretation

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pp", "tl", "pl"]})
