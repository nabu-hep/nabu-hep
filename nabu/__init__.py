import json
from pathlib import Path

import equinox as eqx

from nabu.likelihood.nflow_likelihood import MAFLikelihood

from ._version import __version__
from .goodness_of_fit import Histogram
from .maf import masked_autoregressive_flow
from .plot import chi2_analysis
from .train import fit
from .transform_base import PosteriorTransform


def load_from_file(filename: str, random_seed: int = 0):

    nabu_file = Path(filename)
    assert nabu_file.suffix == ".nabu" and nabu_file.exists(), "Invalid input format."

    with open(str(nabu_file), "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model_type = hyperparams["model_type"]
        config = hyperparams["model_config"]
        model_id = config.pop("type")
        standardisation = hyperparams["standardisation"]
        transform = PosteriorTransform(
            name=standardisation["name"], **standardisation.get("kwargs", {})
        )
        notes = hyperparams.get("notes", "")
        version = hyperparams["version"]

        if model_type == "nflow" and model_id == "maf":
            likelihood = MAFLikelihood(
                **config,
                posterior_transform=transform,
                notes=notes,
                version=version,
                random_seed=random_seed,
            )
            likelihood.model = eqx.tree_deserialise_leaves(f, likelihood.model)

        else:
            raise NotImplementedError(f"{model_type} currently not implemented")

    return likelihood
