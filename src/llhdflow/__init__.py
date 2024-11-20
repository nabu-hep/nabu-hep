from ._version import __version__
from .maf import masked_autoregressive_flow
from .train import fit
from .goodness_of_fit import Histogram
from .plot import chi2_analysis
from .likelihood import Likelihood


def version():
    """Retreive version info"""
    return __version__
