from ._version import __version__
from .likelihood import Likelihood
from .transform_base import *


def version() -> str:
    """Retreive nabu version"""
    return __version__
