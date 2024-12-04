import json
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path

import equinox as eqx
import jax.random as jr
import numpy as np
from jax.tree_util import tree_structure
from scipy.stats import chi2

from .goodness_of_fit import Histogram

__all__ = ["Likelihood"]


def __dir__():
    return __all__


# pylint: disable=import-outside-toplevel


class Likelihood(ABC):
    """Base Likelihood Definition"""

    model_type: str = "base"
    """Model type"""

    __slots__ = ["_model", "_posterior_transform"]

    def __init__(
        self,
        model,
        posterior_transform,
    ):
        self._model = model
        self._posterior_transform = posterior_transform

    @property
    def model(self):
        """Retreive the likelihood model"""
        return self._model

    @model.setter
    def model(self, model):
        assert hasattr(model, "log_prob") and hasattr(model, "sample"), "Invalid model"
        assert tree_structure(self._model) == tree_structure(
            model
        ), "Likelihood definition does not match"
        self._model = model

    @property
    def transform(self):
        """Retreive Posterior transformer"""
        return self._posterior_transform

    @transform.setter
    def transform(self, ptransform):
        from nabu.transform_base import PosteriorTransform

        assert isinstance(ptransform, PosteriorTransform), "Invalid input"
        self._posterior_transform = ptransform

    @abstractmethod
    def to_dict(self) -> dict:
        """convert model to dictionary"""

    @abstractmethod
    def inverse(self):
        """return inverse of the model"""

    def serialise(self) -> dict:
        """
        _summary_

        Returns:
            ``dict``:
            _description_
        """
        return {
            "model_type": self.model_type,
            "model": self.to_dict(),
            "posterior_transform": self.transform.to_dict(),
        }

    def sample(self, size: int, random_seed: int = 0) -> np.ndarray:
        """
        Generate samples from the underlying normalising flow

        Args:
            size (``int``): number of samples to be generated
            random_seed (``int``, default ``0``): random seed

        Returns:
            ``np.ndarray``:
            generated samples
        """
        return np.array(
            self.transform.backward(self.model.sample(jr.key(random_seed), (size,)))
        )

    def log_prob(self, x: np.array) -> np.ndarray:
        """Compute log-probability"""
        return np.array(self.model.log_prob(self.transform.backward(x)))

    def save(self, filename: str) -> None:
        """
        Save likelihood

        Args:
            filename (``str``): file name
        """
        path = Path(filename)
        if path.suffix != ".nabu":
            path = path.with_suffix(".nabu")
        config = self.serialise()
        config.update({"version": version("nabu")})

        with open(str(path), "wb") as f:
            hyperparam_str = json.dumps(self.serialise())
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.model)

    @staticmethod
    def load(filename: str, random_seed: int):
        """
        Load likelihood from file

        Args:
            filename (``str``): file name
            random_seed (``int``): random seed for initialisation

        Returns:
            ``Likelihood``:
            Likelihood object
        """
        from nabu.flow import get_bijector, get_flow
        from nabu.transform_base import PosteriorTransform

        nabu_file = Path(filename)
        assert nabu_file.suffix == ".nabu" and nabu_file.exists(), "Invalid input format."

        with open(str(nabu_file), "rb") as f:
            likelihood_definition = json.loads(f.readline().decode())
            assert (
                likelihood_definition["model_type"] == "flow"
            ), "Given file does not contain a flow type likelihood"
            flow_id, flow_kwargs = list(*likelihood_definition["model"].items())

            if flow_kwargs.get("transformer", None) is not None:
                transformer, transformer_kwargs = list(
                    *flow_kwargs["transformer"].items()
                )
                flow_kwargs["transformer"] = get_bijector(transformer)(
                    **transformer_kwargs
                )

            flow_kwargs["random_seed"] = random_seed
            likelihood = get_flow(flow_id)(**flow_kwargs)
            likelihood.model = eqx.tree_deserialise_leaves(f, likelihood.model)

            ptrans_def, kwargs = list(
                *likelihood_definition["posterior_transform"].items()
            )
            likelihood.transform = PosteriorTransform(ptrans_def, **kwargs)

        return likelihood
