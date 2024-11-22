from collections.abc import Sequence

import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from jax.nn import relu, sigmoid, tanh

from nabu.maf import masked_autoregressive_flow
from nabu.transform_base import PosteriorTransform

from .likelihood_base import LikelihoodBase


class MAFLikelihood(LikelihoodBase):
    """
    Likelihood constructed with Masked Autoregressive Flow

    Args:
        dim (``int``): _description_
        transformer (``str``, default ``"affine"``): _description_
        activation (``str``, default ``"relu"``): _description_
        nn_width (``int``, default ``64``): _description_
        nn_depth (``int``, default ``2``): _description_
        flow_layers (``int``, default ``8``): _description_
        permutation (``Sequence[int]``, default ``None``): _description_
        random_seed (``int``, default ``0``): _description_
        posterior_transform (``PosteriorTransform``, default ``None``): _description_
        transformer_kwargs (``dict``, default ``None``): _description_
    """

    model_type: str = "nflow"
    __slots__ = ["_meta"]

    def __init__(
        self,
        dim: int,
        transformer: str = "affine",
        activation: str = "relu",
        nn_width: int = 64,
        nn_depth: int = 2,
        flow_layers: int = 8,
        permutation: Sequence[int] = None,
        random_seed: int = 0,
        posterior_transform: PosteriorTransform = None,
        transformer_kwargs: dict = None,
        **kwargs,
    ):
        transformer_kwargs = transformer_kwargs or {}
        self._meta = dict(
            type="maf",
            dim=dim,
            transformer=transformer,
            activation=activation,
            nn_width=nn_width,
            nn_depth=nn_depth,
            flow_layers=flow_layers,
            permutation=permutation or list(reversed(range(dim))),
            transformer_kwargs=transformer_kwargs,
        )
        transformer = {
            "affine": None,
            "rqs": RationalQuadraticSpline(
                knots=transformer_kwargs.get("knots", 6),
                interval=transformer_kwargs.get("interval", 4),
            ),
        }[transformer]

        activation = {"relu": relu, "sigmoid": sigmoid, "tanh": tanh}[activation]

        model = masked_autoregressive_flow(
            jr.key(random_seed),
            base_dist=Normal(jnp.zeros(dim)),
            transformer=transformer,
            nn_width=nn_width,
            nn_depth=nn_depth,
            flow_layers=flow_layers,
            invert=True,
            nn_activation=activation,
            permutation=permutation or list(reversed(range(dim))),
        )

        super().__init__(
            model=model,
            posterior_transform=posterior_transform or PosteriorTransform(),
            **kwargs,
        )

    def serialise(self) -> dict:
        return self._meta
