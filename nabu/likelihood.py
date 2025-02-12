import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.tree_util import tree_structure
from scipy.stats import chi2, kstest

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
    def inverse(self) -> Callable:
        """return inverse of the model"""

    def compute_inverse(self, x: np.ndarray) -> np.ndarray:
        """Compute inverse likelihood for a given dataset"""
        return np.array(self.inverse()(jnp.array(self.transform.backward(x))))

    @abstractmethod
    def fit_to_data(self, *args, **kwargs) -> dict[str, list[float]]:
        """Fit likelihood to given dataset"""

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

    def chi2(self, x: np.ndarray) -> np.ndarray:
        """
        Compute chi^2 for

        Args:
            x (``np.ndarray``): input data

        Returns:
            ``np.ndarray``:
            chi2 for given x. shape (N,dof)
        """
        x = np.expand_dims(x, 0) if len(x.shape) == 1 else x
        return np.sum(self.compute_inverse(x) ** 2, axis=1)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Compute the cumulative density function at x shape (N,dof)"""
        return chi2.cdf(self.chi2(x), df=x.shape[-1])

    def kstest_pvalue(self, test_dataset: np.ndarray) -> float:
        """Compute p-value for Kolmogorov-Smirnov test"""
        return kstest(
            self.chi2(test_dataset), cdf=partial(chi2.cdf, df=test_dataset.shape[-1])
        ).pvalue

    def goodness_of_fit(
        self,
        test_dataset: np.ndarray,
        prob_per_bin: float = 0.1,
    ) -> Histogram:
        """
        Construct binned and unbinned goodness of fit test.

        Args:
            test_dataset (``np.ndarray``): test dataset, shape `(N, DoF)`
            prob_per_bin (``float``, default ``0.1``): probability of yields in each bin

        Returns:
            ``Histogram``:
            Goodness of fit base
        """
        dim = test_dataset.shape[-1]
        bins = chi2.ppf(
            np.linspace(0.0, 1.0, int(np.ceil(1.0 / prob_per_bin)) + 1), df=dim
        )
        return Histogram(dim=dim, bins=bins, vals=self.chi2(test_dataset))

    def is_valid(self, test_data: np.ndarray, threshold: float = 0.03) -> bool:
        """
        Check if the likelihood is valid for a given dataset.

        Args:
            test_data (``np.ndarray``): Test dataset
            threshold (``float``, default ``0.03``): p-value threshold.

        Returns:
            ``bool``:
            Returns `True` if the model passes the threshold.
        """
        hist: Histogram = self.goodness_of_fit(test_dataset=test_data, prob_per_bin=0.1)
        return all(np.greater_equal([hist.residuals_pvalue, hist.kstest_pval], threshold))

    @abstractmethod
    def to_dict(self) -> dict:
        """convert model to dictionary"""

    def serialise(self) -> dict:
        """
        Serialise the underlying model

        Returns:
            ``dict``:
            Description of the model
        """
        assert self.model_type != "base", "Invalid model type"
        return {
            "model_type": self.model_type,
            "model": self.to_dict(),
            "posterior_transform": self.transform.to_dict(),
        }

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
            hyperparam_str = json.dumps(config, cls=NumpyEncoder)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.model)

    @staticmethod
    def load(filename: str, random_seed: int = np.random.randint(0, high=999999999999)):
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
                for key in transformer_kwargs:
                    if isinstance(transformer_kwargs[key], list):
                        transformer_kwargs[key] = tuple(transformer_kwargs[key])
                flow_kwargs["transformer"] = get_bijector(transformer)(
                    **transformer_kwargs
                )

            flow_kwargs["random_seed"] = random_seed
            likelihood = get_flow(flow_id)(**flow_kwargs)
            likelihood.model = eqx.tree_deserialise_leaves(f, likelihood.model)

            if likelihood_definition["posterior_transform"] != {}:
                ptrans_def, kwargs = list(
                    *likelihood_definition["posterior_transform"].items()
                )
                likelihood.transform = PosteriorTransform(ptrans_def, **kwargs)
            else:
                likelihood.transform = PosteriorTransform()

        return likelihood


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)
