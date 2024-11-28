import json
import os
from pathlib import Path

import equinox as eqx

from ._version import __version__
from .flow.bijectors import *
from .flow.flows import *
from .transform_base import PosteriorTransform

RANDOM_SEED = 42

if os.environ.get("NABU_RANDOM_SEED", False) is not False:
    RANDOM_SEED = os.environ["NABU_RANDOM_SEED"]


def version() -> str:
    """Retreive nabu version"""
    return __version__


def load_from_file(filename: str, random_seed: int = RANDOM_SEED):
    """
    data structure:

    {
        "model_type": "flow",
        "model: {"<flow_name>": {**kwargs} }, # includes model definition
        "posterior_transform": {"<transform-name>": {**kwargs} }, # includes posterior transform definition
        "version": "x.x.x",
    }

    Args:
        filename (``str``): _description_
        random_seed (``int``): _description_

    Returns:
        ``_type_``:
        _description_
    """
    nabu_file = Path(filename)
    assert nabu_file.suffix == ".nabu" and nabu_file.exists(), "Invalid input format."

    with open(str(nabu_file), "rb") as f:
        likelihood_definition = json.loads(f.readline().decode())
        assert (
            likelihood_definition["model_type"] == "flow"
        ), "Given file does not contain a flow type likelihood"
        flow_id, flow_kwargs = list(*likelihood_definition["model"].items())

        if flow_kwargs.get("transformer", None) is not None:
            transformer, transformer_kwargs = list(*flow_kwargs["transformer"].items())
            flow_kwargs["transformer"] = get_bijector(transformer)(**transformer_kwargs)

        flow_kwargs["random_seed"] = random_seed
        likelihood = get_flow(flow_id)(**flow_kwargs)
        likelihood.model = eqx.tree_deserialise_leaves(f, likelihood.model)

        ptrans_def, kwargs = list(*likelihood_definition["posterior_transform"].items())
        likelihood.transform = PosteriorTransform(ptrans_def, **kwargs)

    return likelihood
