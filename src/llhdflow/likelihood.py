from typing import Callable, Dict, Text

import jax
import jax.random as jr
import numpy as np
from flowjax.distributions import Transformed
from scipy.stats import chi2

from .build_flow import build_flow, create_sampler
from .goodness_of_fit import Histogram


class Likelihood:
    def __init__(self):
        self._flow: Transformed = None
        self._posterior_transform: Callable[
            [np.ndarray], np.ndarray
        ] = lambda x: x  # null transformer
        self._model_config: Dict = {}

    def set_flow(self, flow: Transformed) -> None:
        assert isinstance(flow, Transformed), "Invalid input"
        self._flow = flow

    def set_posterior_transform(
        self, posterior_transform: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        assert callable(posterior_transform), "Invalid input"
        self._posterior_transform = posterior_transform

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
        assert self._flow is not None, "Likelihood has not been initialised"
        return np.array(
            self._posterior_transform(self._flow.sample(jr.key(random_seed), (size,)))
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
        llhd.set_flow(flow)

        if posterior_transform_file is not None:
            _, ptrans = create_sampler(
                flow,
                posterior_transform_file=posterior_transform_file,
                transformer_name=transformer_name,
                return_posterior_transformer=True,
            )
            llhd.set_posterior_transform(ptrans)

        return llhd

    @classmethod
    def load_from_zenodo(cls, zenodo_id: int):
        """under construction"""

    def goodness_of_fit(self, test_data: np.ndarray) -> Histogram:
        gauss_test = np.array(
            jax.vmap(self._flow.bijection.inverse, in_axes=0)(test_data)
        )
        chi2_test = np.sum(gauss_test**2, axis=1)
        dim = test_data.shape[1]
        bins = chi2.ppf(np.linspace(0.0, 1.0, int(np.ceil(1.0 / 0.1)) + 1), df=dim)
        return Histogram(dim=dim, bins=bins, vals=chi2_test)
