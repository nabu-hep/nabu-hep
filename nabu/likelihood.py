from pathlib import Path
from typing import Dict, Text

import jax
import jax.random as jr
import numpy as np
from flowjax.distributions import Transformed
from scipy.stats import chi2

from .build_flow import build_flow, create_sampler
from .goodness_of_fit import Histogram
from .transform_base import PosteriorTransformBase


class Likelihood:
    def __init__(self):
        self._model: Transformed = None
        self._posterior_transform: PosteriorTransformBase = PosteriorTransformBase()
        self._model_config: Dict = {}

    @property
    def model(self) -> Transformed:
        """Retreive the likelihood model"""
        assert self._model is not None, "Likelihood model has not been initialised"
        return self._model

    @model.setter
    def model(self, model: Transformed) -> None:
        assert isinstance(model, Transformed), "Invalid input"
        self._model = model

    @property
    def posterior_transform(self) -> PosteriorTransformBase:
        """Retreive Posterior transformer"""
        return self._posterior_transform

    @posterior_transform.setter
    def posterior_transform(self, transform: PosteriorTransformBase) -> None:
        assert isinstance(transform, PosteriorTransformBase), "Invalid input"
        self._posterior_transform = transform

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
        return self._posterior_transform._backward(
            self.model.sample(jr.key(random_seed), (size,))
        )

    @classmethod
    def load_from_file(
        cls,
        config_file: Text,
        model_file: Text,
        random_seed: int = 0,
        posterior_transform_file: Text = None,
        transformer_name: Text = "PosteriorTransform",
    ):
        """
        Load likelihood from file

        Args:
            config_file (``Text``): model configuration file
            model_file (``Text``): saved model file
            random_seed (``int``, default ``0``): random seed
            posterior_transform_file (``Text``, default ``None``): file that defines
                posterior transformation
            transformer_name (``Text``, default ``"PosteriorTransform"``): name of the class
                or the function that performs transformation
        """
        flow = build_flow(config_file, model_file, random_seed)
        llhd = cls()
        llhd.model = flow

        if posterior_transform_file is not None:
            _, ptrans = create_sampler(
                flow,
                posterior_transform_file=posterior_transform_file,
                transformer_name=transformer_name,
                return_posterior_transformer=True,
            )
            llhd.posterior_transform = ptrans

        return llhd

    @classmethod
    def load_from_zenodo(cls, zenodo_id: int):
        """under construction"""

    def goodness_of_fit(self, test_data: np.ndarray) -> Histogram:

        gauss_test = np.array(
            jax.vmap(self.model.bijection.inverse, in_axes=0)(
                self.posterior_transform._forward(test_data)
            )
        )
        chi2_test = np.sum(gauss_test**2, axis=1)
        dim = test_data.shape[1]
        bins = chi2.ppf(np.linspace(0.0, 1.0, int(np.ceil(1.0 / 0.1)) + 1), df=dim)
        return Histogram(dim=dim, bins=bins, vals=chi2_test)
