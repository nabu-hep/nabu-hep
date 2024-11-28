import json
from abc import ABC, abstractmethod
from pathlib import Path

import equinox as eqx
import jax.random as jr
import numpy as np

from nabu.transform_base import PosteriorTransform
from nabu import __version__


class LikelihoodBase(ABC):
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
        self._model = model

    @property
    def transform(self):
        """Retreive Posterior transformer"""
        return self._posterior_transform

    @transform.setter
    def transform(self, ptransform: PosteriorTransform):
        assert isinstance(ptransform, PosteriorTransform), "Invalid input"
        self._posterior_transform = ptransform

    @abstractmethod
    def to_dict(self) -> dict:
        """convert model to dictionary"""

    @abstractmethod
    def inverse(self):
        """return inverse of the model"""

    def serialise(self):
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
        return self.transform.backward(self.model.sample(jr.key(random_seed), (size,)))

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
        config.update({"version": __version__})

        with open(str(path), "wb") as f:
            hyperparam_str = json.dumps(self.serialise())
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.model)
