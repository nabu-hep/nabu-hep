import json
from abc import ABC
from pathlib import Path

import equinox as eqx
import jax.random as jr
import numpy as np


class LikelihoodBase(ABC):
    model_type: str = "__unknown_model__"
    __slots__ = ["_model", "_posterior_transform", "notes", "version"]

    def __init__(
        self,
        model,
        posterior_transform,
        notes: str = None,
        version: str = "0.0.1",
    ):
        self._model = model
        self._posterior_transform = posterior_transform
        self.notes = notes
        self.version = version

    @property
    def model(self):
        """Retreive the likelihood model"""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def transform(self):
        """Retreive Posterior transformer"""
        return self._posterior_transform

    def serialise(self) -> dict:
        """Serialise the likelihood"""
        raise NotImplementedError

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
        return self.model.log_prob(self.transform.backward(x))

    def save(self, filename: str) -> None:
        """
        Save likelihood

        Args:
            filename (``str``): file name
        """
        path = Path(filename)
        if path.suffix != ".nabu":
            path = path.with_suffix(".nabu")
        config = {"model_type": self.model_type}
        config.update({"model_config": self.serialise()})
        config.update({"standardisation": self.transform.serialise()})
        config.update({"notes": self.notes, "version": self.version})

        with open(str(path), "wb") as f:
            hyperparam_str = json.dumps(config)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.model)
