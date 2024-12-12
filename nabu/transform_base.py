from typing import Any, Literal, Tuple

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from .dalitz_utils import dalitz_to_square, square_to_dalitz

Array = Any

__all__ = [
    "PosteriorTransform",
    "standardise_between_negone_and_one",
    "standardise_between_zero_and_one",
    "standardise_median_quantile",
    "standardise_mean_std",
]


def __dir__():
    return __all__


class PosteriorTransform:
    """Base class to handle transformations"""

    def __init__(
        self,
        name: Literal["mean_std", "dalitz", "min-max", "user-defined"] = None,
        **kwargs,
    ):
        """
        Definition of posterior transformation

        Args:
            name (``Text``, default ``None``): Identifier of the implementation
                * ``mean_std``: this input expects user to provide a list of mean values
                    and a list of standard deviations for the features
                    * ``mean``: mean values of the features
                    * ``std``: standard deviations
                * ``dalitz``: this identifier initialise conversion from a dalitz distribution to
                    square distribution within [0,1]
                    * ``md``: mother particle mass
                    * ``ma``: first daughter particle
                    * ``mb``: second daogter particle
                    * ``mc``: third daugher particle
                * ``user-defined``: this input expect user to define forward and backward functions
                    * ``forward``: a function that takes array as input and returns an array
                        with same dimensionality. This is used to convert features to the traning basis
                    * ``backward``: a function that takes array as input and returns an array
                        with same dimensionality. This is used to convert features to the physical basis
        """
        if name == "mean_std":
            mean, scale = jnp.array(kwargs["mean"]), jnp.array(kwargs["std"])
            self._metadata = {"mean_std": {"mean": mean.tolist(), "std": scale.tolist()}}

            def forward(x):
                return (x - mean) / scale

            def backward(x):
                return x * scale + mean

        elif name == "dalitz":
            md, ma, mb, mc = (
                kwargs["md"],
                kwargs["ma"],
                kwargs["mb"],
                kwargs["mc"],
            )
            self._metadata = {"dalitz": {"md": md, "ma": ma, "mb": mb, "mc": mc}}

            def forward(x):
                return dalitz_to_square(x, md, ma, mb, mc)

            def backward(x):
                return square_to_dalitz(x, md, ma, mb, mc)

        elif name == "min-max":
            minimum, scale = jnp.array(kwargs["minimum"]), jnp.array(kwargs["scale"])

            self._metadata = {
                "min-max": {"minimum": minimum.tolist(), "scale": scale.tolist()}
            }

            def forward(x):
                return (x - minimum) / scale

            def backward(x):
                return x * scale + minimum

        elif name is None:
            forward = lambda x: x
            backward = lambda x: x
            self._metadata = {}

        elif name == "user-defined":
            forward = kwargs["forward"]
            backward = kwargs["backward"]
            assert all(
                callable(f) for f in [forward, backward]
            ), "Invalid function definition"
            self._metadata = {}

        else:
            raise NotImplementedError(f"{name} currently not implemented")

        self._forward = forward
        self._backward = backward

    def to_dict(self):
        return self._metadata

    @classmethod
    def from_mean_std(cls, mean: list[float], std: list[float]):
        """Generate transform from mean and standard deviation"""
        return cls("mean_std", mean=mean, std=std)

    @classmethod
    def from_min_max(cls, minimum: list[float], scale: list[float]):
        """Generate transform from min-max"""
        return cls("min-max", minimum=minimum, scale=scale)

    @classmethod
    def from_dalitz(cls, md: float, ma: float, mb: float, mc: float):
        """Generate transform from dalitz conversion"""
        return cls("dalitz", md=md, ma=ma, mb=mb, mc=mc)

    def forward(self, x: Array) -> Array:
        """
        Take unmodified input and transform it.
        Output of this function will be fed into the ML model

        Args:
            x (``Array``): input data

        Returns:
            ``Array``:
            Transformed data
        """
        return jnp.array(self.forward(x))

    def backward(self, y: Array) -> Array:
        """
        Take transformed data and convert it to original.
        Output of this fuction is returned to the user.

        Args:
            y (``Array``): transformed data

        Returns:
            ``Array``:
            Original data
        """
        return jnp.array(self._backward(y))


def standardise_between_zero_and_one(
    data: np.ndarray,
) -> tuple[PosteriorTransform, np.ndarray]:
    """
    standardise data between [0,1]

    Args:
        data (``np.ndarray``): dataset with shape (N, M). N=number of data, M=number of features

    Returns:
        ``Tuple[PosteriorTransform, np.ndarray]``:
        Transform function and standardised data.
    """
    mean = np.min(data, axis=0)
    scale = np.max(data, axis=0) - np.min(data, axis=0)
    return PosteriorTransform("mean_std", mean=mean, std=scale), (data - mean) / scale


def standardise_between_negone_and_one(
    data: np.ndarray,
) -> tuple[PosteriorTransform, np.ndarray]:
    """
    standardise data between [-1,1]

    Args:
        data (``np.ndarray``): dataset with shape (N, M). N=number of data, M=number of features

    Returns:
        ``Tuple[PosteriorTransform, np.ndarray]``:
        Transform function and standardised data.
    """
    mn = np.min(data, axis=0)
    scale = (np.max(data, axis=0) - mn) / 2.0
    mean = mn + scale
    return PosteriorTransform("mean_std", mean=mean, std=scale), (data - mean) / scale


def standardise_median_quantile(
    data: np.ndarray,
) -> tuple[PosteriorTransform, np.ndarray]:
    """
    _summary_

    Args:
        data (``np.ndarray``): dataset with shape (N, M). N=number of data, M=number of features

    Returns:
        ``Tuple[PosteriorTransform, np.ndarray]``:
        Transform function and standardised data.
    """
    q = (1 - (norm.cdf(1) - norm.cdf(-1))) / 2
    mean = np.median(data, axis=0)
    scale = np.quantile(data, 1 - q, axis=0) - np.quantile(data, q, axis=0)
    return PosteriorTransform("mean_std", mean=mean, std=scale), (data - mean) / scale


def standardise_mean_std(
    data: np.ndarray,
) -> tuple[PosteriorTransform, np.ndarray]:
    """
    _summary_

    Args:
        data (``np.ndarray``): dataset with shape (N, M). N=number of data, M=number of features

    Returns:
        ``Tuple[PosteriorTransform, np.ndarray]``:
        Transform function and standardised data.
    """
    mean = np.mean(data, axis=0)
    scale = np.std(data, axis=0)
    return PosteriorTransform("mean_std", mean=mean, std=scale), (data - mean) / scale
