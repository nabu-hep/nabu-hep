"""
This file is outdated and replaced with flow/flows.py
keeping here until fit-to-data is modified
"""
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import (
    AbstractBijection,
    Affine,
    Chain,
    Flip,
    Invert,
    MaskedAutoregressive,
    Permute,
    Scan,
)
from flowjax.distributions import AbstractDistribution, Transformed
from flowjax.utils import inv_softplus
from flowjax.wrappers import Parameterize
from jaxtyping import PRNGKeyArray

__all__ = ["masked_autoregressive_flow"]


def __dir__():
    return __all__


def _affine_with_min_scale(min_scale: float = 1e-2) -> Affine:
    scale = Parameterize(
        lambda x: jax.nn.softplus(x) + min_scale, inv_softplus(1 - min_scale)
    )
    return eqx.tree_at(where=lambda aff: aff.scale, pytree=Affine(), replace=scale)


def _add_default_permute(
    bijection: AbstractBijection,
    dim: int,
    key: PRNGKeyArray,
    permutation: jnp.array = None,
):
    if dim == 1:
        return bijection
    if dim == 2:
        return Chain([bijection, Flip((dim,))]).merge_chains()

    if permutation is None:
        perm = Permute(jax.random.permutation(key, jnp.arange(dim)))
    else:
        perm = Permute(permutation)
    return Chain([bijection, perm]).merge_chains()


def masked_autoregressive_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    transformer: AbstractBijection = None,
    cond_dim: int = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jax.nn.relu,
    invert: bool = True,
    permutation: jnp.array = None,
) -> Transformed:
    """
    Parameterises a transformer bijection with an autoregressive neural network.
    Refs: https://arxiv.org/abs/1606.04934; https://arxiv.org/abs/1705.07057v4.

    based on flowjax construction with minor changes such as permutation option.

    Args:
        key (``PRNGKeyArray``): random seed
        base_dist (``AbstractDistribution``): base distribution
        transformer (``AbstractBijection``, default ``None``): _description_
        cond_dim (``int``, default ``None``): _description_
        flow_layers (``int``, default ``8``): _description_
        nn_width (``int``, default ``50``): _description_
        nn_depth (``int``, default ``1``): _description_
        nn_activation (``Callable``, default ``jax.nn.relu``): _description_
        invert (``bool``, default ``True``): _description_
        return_bijection (``bool``, default ``False``): _description_
        permutation (``jnp.array``, default ``None``): _description_

    Returns:
        ``Transformed``:
        _description_
    """

    if transformer is None:
        transformer = _affine_with_min_scale()

    dim = base_dist.shape[-1]

    def make_layer(key):  # masked autoregressive layer + permutation
        bij_key, perm_key = jax.random.split(key)
        bijection = MaskedAutoregressive(
            key=bij_key,
            transformer=transformer,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        return _add_default_permute(bijection, dim, perm_key, permutation)

    keys = jax.random.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)
