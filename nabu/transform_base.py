from typing import Any, Literal

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
        name: Literal["shift-scale", "dalitz", "user-defined"] = None,
        **kwargs,
    ):
        """
        Definition of posterior transformation

        Args:
            name (``Text``, default ``None``): Identifier of the implementation
                * ``shift-scale``:
                    * ``shift``: shift the dataset
                    * ``scale``: scale the dataset
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
        if name == "shift-scale":
            mean, scale = jnp.array(kwargs["shift"]), jnp.array(kwargs["scale"])
            log_axes = kwargs.get("log_axes", None)
            log_shift = kwargs.get("log_shift", None)
            log_scale = kwargs.get("log_scale", None)
            if log_shift is not None:
                log_shift = np.array(log_shift)
            if log_scale is not None:
                log_scale = np.array(log_scale)
            self._metadata = {
                "shift-scale": {
                    "shift": mean.tolist(),
                    "scale": scale.tolist(),
                    "log_axes": log_axes,
                    "log_shift": log_shift.tolist() if log_shift is not None else None,
                    "log_scale": log_scale.tolist() if log_scale is not None else None,
                }
            }

            def forward(x):
                x_new = (x - mean) / scale
                if log_axes is not None:
                    x_new = np.array(x_new)
                    x_new[:, log_axes] = np.log(x_new[:, log_axes])
                    x_new[:, log_axes] = (x_new[:, log_axes] - log_shift) / log_scale
                return x_new

            def backward(x):
                if log_axes is not None:
                    x = np.array(x)
                    x[:, log_axes] = x[:, log_axes] * log_scale + log_shift
                    x[:, log_axes] = np.exp(x[:, log_axes])
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
    def from_shift_scale(
        cls,
        shift: list[float],
        scale: list[float],
        log_axes: list[int] = None,
        log_shift=None,
        log_scale=None,
    ):
        """Shift and scale the dataset"""
        return cls(
            "shift-scale",
            shift=shift,
            scale=scale,
            log_axes=log_axes,
            log_shift=log_shift,
            log_scale=log_scale,
        )

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
        return jnp.array(self._forward(x))

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


def _check_weights(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Validate and coerce per-sample weights for the weighted standardisers.

    Args:
        data (``np.ndarray``): dataset with shape (N, M).
        weights (``np.ndarray``): per-sample weights of length N.

    Returns:
        ``np.ndarray``:
        The weights coerced to a float array.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError(
            f"weights must be a 1D array of length {len(data)}, got shape {weights.shape}"
        )
    if len(weights) != len(data):
        raise ValueError(
            f"data and weights must have the same length, got {len(data)} and "
            f"{len(weights)}"
        )
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    if not weights.sum() > 0.0:
        raise ValueError("the total weight must be positive")
    return weights


def _weighted_quantile(data: np.ndarray, quantiles, weights: np.ndarray) -> np.ndarray:
    r"""
    Weighted quantiles computed per feature via linear interpolation of the
    weighted ECDF.

    Uses Hazen plotting positions :math:`p_i = (C_i - w_i / 2) / \sum_j w_j`, where
    :math:`C_i` is the cumulative weight up to the (sorted) sample :math:`i`. With
    uniform weights this matches ``numpy.quantile(..., method="hazen")``; in particular
    the median (``q = 0.5``) agrees with ``numpy.median``. Samples with zero weight are
    dropped. Requested quantiles outside the support are clamped to the extreme samples.

    Args:
        data (``np.ndarray``): dataset with shape (N,) or (N, M).
        quantiles (``float`` or ``np.ndarray``): quantile(s) in ``[0, 1]``.
        weights (``np.ndarray``): per-sample weights of length N.

    Returns:
        ``np.ndarray``:
        The interpolated quantile(s). For 2D ``data`` the feature axis is last, so a
        length-K ``quantiles`` yields shape ``(K, M)`` and a scalar yields shape ``(M,)``.
    """
    data = np.asarray(data, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    mask = weights > 0.0
    data, weights = data[mask], weights[mask]
    sumw = weights.sum()

    def _col(col: np.ndarray) -> np.ndarray:
        order = np.argsort(col, kind="mergesort")
        v, w = col[order], weights[order]
        # Hazen plotting positions; strictly increasing since all weights are positive.
        p = (np.cumsum(w) - 0.5 * w) / sumw
        return np.interp(quantiles, p, v)

    if data.ndim == 1:
        return _col(data)
    return np.stack([_col(data[:, j]) for j in range(data.shape[1])], axis=-1)


def standardise_dalitz(
    data: np.ndarray,
    md: float = 1e-3,
    ma: float = 1e-3,
    mb: float = 1e-3,
    mc: float = 1e-3,
) -> tuple[PosteriorTransform, np.ndarray]:
    """
    Convert dalitz plot cartesisan coordinates, D -> A B C

    Args:
        data (``np.ndarray``): input data to be standardised
        md (``float``): mother particle [GeV]
        ma (``float``): daughter particle [GeV]
        mb (``float``): daughter particle [GeV]
        mc (``float``): daughter particle [GeV]

    Returns:
        ``tuple[PosteriorTransform, np.ndarray]``:
        Transformer and transformed data.
    """
    return (
        PosteriorTransform.from_dalitz(md, ma, mb, mc),
        dalitz_to_square(data, md, ma, mb, mc),
    )


def standardise_between_zero_and_one(
    data: np.ndarray, log_axes: list[int] = None, eps: float = 1e-6
) -> tuple[PosteriorTransform, np.ndarray]:
    """
    standardise data between [0,1]

    Args:
        data (``np.ndarray``): dataset with shape (N, M). N=number of data, M=number of features

    Returns:
        ``Tuple[PosteriorTransform, np.ndarray]``:
        Transform function and standardised data.
    """
    log_shift, log_scale = None, None
    mean = np.min(data, axis=0) - eps
    scale = np.max(data, axis=0) - np.min(data, axis=0)

    new_data = (data - mean) / scale
    if log_axes is not None:
        new_data[:, log_axes] = np.log(new_data[:, log_axes])
        log_shift = np.min(new_data[:, log_axes], axis=0)
        log_scale = np.max(new_data[:, log_axes], axis=0) - log_shift
        new_data[:, log_axes] = (new_data[:, log_axes] - log_shift) / log_scale

    return (
        PosteriorTransform.from_shift_scale(
            shift=mean,
            scale=scale,
            log_axes=log_axes,
            log_shift=log_shift,
            log_scale=log_scale,
        ),
        new_data,
    )


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
    return (
        PosteriorTransform.from_shift_scale(shift=mean, scale=scale),
        (data - mean) / scale,
    )


def standardise_median_quantile(
    data: np.ndarray,
    weights: np.ndarray = None,
) -> tuple[PosteriorTransform, np.ndarray]:
    r"""
    Shift and scale the data using median and :math:`1\sigma` quantile

    Args:
        data (``np.ndarray``): dataset with shape (N, M). N=number of data, M=number of features
        weights (``np.ndarray``, default ``None``): per-sample weights of length N. If
            ``None``, the unweighted median and quantiles are used. Otherwise the shift is
            the weighted median and the scale is the weighted :math:`1\sigma` inter-quantile
            range, so the embedded transform centres and scales a weighted target correctly.

    Returns:
        ``Tuple[PosteriorTransform, np.ndarray]``:
        Transform function and standardised data.
    """
    q = (1 - (norm.cdf(1) - norm.cdf(-1))) / 2
    if weights is None:
        mean = np.median(data, axis=0)
        scale = np.quantile(data, 1 - q, axis=0) - np.quantile(data, q, axis=0)
    else:
        weights = _check_weights(data, weights)
        lo, mean, hi = _weighted_quantile(data, [q, 0.5, 1 - q], weights)
        scale = hi - lo
    return (
        PosteriorTransform.from_shift_scale(shift=mean, scale=scale),
        (data - mean) / scale,
    )


def standardise_mean_std(
    data: np.ndarray,
    weights: np.ndarray = None,
) -> tuple[PosteriorTransform, np.ndarray]:
    """
    Shift and scale the data using mean and standard deviation

    Args:
        data (``np.ndarray``): dataset with shape (N, M). N=number of data, M=number of features
        weights (``np.ndarray``, default ``None``): per-sample weights of length N. If
            ``None``, the unweighted mean and standard deviation are used. Otherwise the
            shift is the weighted mean and the scale is the (population) weighted
            standard deviation, so the embedded transform centres and scales a weighted
            target correctly.

    Returns:
        ``Tuple[PosteriorTransform, np.ndarray]``:
        Transform function and standardised data.
    """
    if weights is None:
        mean = np.mean(data, axis=0)
        scale = np.std(data, axis=0)
    else:
        weights = _check_weights(data, weights)
        mean = np.average(data, axis=0, weights=weights)
        scale = np.sqrt(np.average((data - mean) ** 2, axis=0, weights=weights))
    return (
        PosteriorTransform.from_shift_scale(shift=mean, scale=scale),
        (data - mean) / scale,
    )
