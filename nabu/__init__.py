import json
import os
from pathlib import Path

import equinox as eqx

from ._version import __version__
from .likelihood import Likelihood
from .transform_base import PosteriorTransform


def version() -> str:
    """Retreive nabu version"""
    return __version__
