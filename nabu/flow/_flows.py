from collections.abc import Callable
from functools import partial, wraps
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import Linear
from flowjax.bijections import (
    AbstractBijection,
    AdditiveCondition,
    Affine,
    BlockAutoregressiveNetwork,
    Chain,
    Coupling,
    Flip,
    Invert,
    LeakyTanh,
    MaskedAutoregressive,
    Permute,
    Planar,
    RationalQuadraticSpline,
    Scan,
    TriangularAffine,
    Vmap,
)
from flowjax.distributions import Normal, Transformed
from flowjax.utils import inv_softplus
from flowjax.wrappers import Parameterize, WeightNormalization
from jax.nn import relu, sigmoid, softmax, softplus, tanh
from jax.nn.initializers import glorot_uniform
from jaxtyping import PRNGKeyArray

from ._flow_likelihood import FlowLikelihood
from ._serialisation_utils import serialise_wrapper, BijectorWrapper

__all__ = [
    "masked_autoregressive_flow",
    "coupling_flow",
    "block_neural_autoregressive_flow",
    "planar_flow",
    "triangular_spline_flow",
    "get_flow",
    "available_flows",
    "register_flow",
    "register_activation",
    "available_activations",
]


def __dir__():
    return __all__


_activation_registry = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "softmax": softmax,
    "softplus": softplus,
}


def available_activations() -> list[str]:
    """retreive available activation functions"""
    return list(_activation_registry.keys())


def _get_activation(activation: str) -> callable:
    """retreive activation"""
    return _activation_registry[activation]


def register_activation(name: str, function: callable) -> None:
    """Register activation function"""
    if callable(function) and name not in _activation_registry:
        _activation_registry.update({name: function})


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


@serialise_wrapper
def masked_autoregressive_flow(
    dim: int,
    transformer: AbstractBijection = None,
    cond_dim: int = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    activation: str = "relu",
    invert: bool = True,
    permutation: list[int] = None,
    random_seed: int = 0,
) -> Transformed:
    """
    Parameterises a transformer bijection with an autoregressive neural network.
    Refs: https://arxiv.org/abs/1606.04934; https://arxiv.org/abs/1705.07057v4.

    .. note::

        Based on flowjax construction with minor changes

    Args:
        key (``PRNGKeyArray``): random seed
        dim (``int``): Dimensions of the base distribution (Normal distribution)
        transformer (``AbstractBijection``, default ``None``): _description_
        cond_dim (``int``, default ``None``): _description_
        flow_layers (``int``, default ``8``): _description_
        nn_width (``int``, default ``50``): _description_
        nn_depth (``int``, default ``1``): _description_
        nn_activation (``str``, default ``jax.nn.relu``): _description_
        invert (``bool``, default ``True``): _description_
        return_bijection (``bool``, default ``False``): _description_
        permutation (``jnp.array``, default ``None``): _description_

    Returns:
        ``Transformed``:
        _description_
    """
    key = jr.key(random_seed)
    activation = _get_activation(activation)
    transformer = transformer or _affine_with_min_scale()
    base_dist = Normal(jnp.zeros(dim))
    if permutation is not None:
        permutation = jnp.array(permutation)

    def make_layer(key):  # masked autoregressive layer + permutation
        bij_key, perm_key = jax.random.split(key)
        bijection = MaskedAutoregressive(
            key=bij_key,
            transformer=transformer,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=activation,
        )
        return _add_default_permute(bijection, dim, perm_key, permutation)

    keys = jax.random.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


@serialise_wrapper
def coupling_flow(
    dim: int,
    transformer: AbstractBijection = None,
    cond_dim: int = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    activation: str = "relu",
    permutation: list[int] = None,
    random_seed: int = 0,
) -> Transformed:
    """
    Create a coupling flow (https://arxiv.org/abs/1605.08803).

    .. note::

        Based on flowjax construction with minor changes

    Args:
        dim (``int``): feature dimensions
        transformer (``AbstractBijection``, default ``None``):
            Bijection to be parameterised by conditioner. Defaults to affine.
        cond_dim (``int``, default ``None``): Dimension of conditioning variables.
        flow_layers (``int``, default ``8``): Number of coupling layers.
        nn_width (``int``, default ``50``): Conditioner hidden layer size.
        activation (``str``, default ``"relu"``): Conditioner activation function.
        permutation (``jnp.array``, default ``None``): Permutation of the features, if
            ``None`` it will be randomly shuffled.
        random_seed (``int``, default ``0``): random seed

    Returns:
        ``Transformed``:
        _description_
    """
    key = jr.key(random_seed)
    activation = _get_activation(activation)
    transformer = transformer or _affine_with_min_scale()
    base_dist = Normal(jnp.zeros(dim))
    if permutation is not None:
        permutation = jnp.array(permutation)

    def make_layer(key):  # coupling layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = Coupling(
            key=bij_key,
            transformer=transformer,
            untransformed_dim=dim // 2,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_activation=activation,
        )
        return _add_default_permute(bijection, dim, perm_key, permutation)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers))
    return Transformed(base_dist, bijection)


@serialise_wrapper
def block_neural_autoregressive_flow(
    dim: int,
    cond_dim: int = None,
    nn_depth: int = 1,
    nn_block_dim: int = 8,
    flow_layers: int = 1,
    activation: str = "sigmoid",
    inverter: Callable = None,
    permutation: list[int] = None,
    random_seed: int = 0,
) -> Transformed:
    """
    Block neural autoregressive flow (BNAF) (https://arxiv.org/abs/1904.04676).

    .. note::

        Based on flowjax construction with minor changes

    Each flow layer contains a
    :class:`~flowjax.bijections.block_autoregressive_network.BlockAutoregressiveNetwork`
    bijection. The bijection does not have an analytic inverse, so must be inverted
    using numerical methods (by default a bisection search). Note that this means
    that only one of ``log_prob`` or ``sample{_and_log_prob}`` can be efficient,
    controlled by the ``invert`` argument.

    Args:
        dim (``int``): _description_
        cond_dim (``int``, default ``None``): Dimension of conditional variables.
        nn_depth (``int``, default ``1``): Number of hidden layers within the networks.
        nn_block_dim (``int``, default ``8``): Block size. Hidden layer width is ``dim*nn_block_dim``.
        flow_layers (``int``, default ``1``): Number of BNAF layers.
        activation (``str``, default ``"relu"``): Activation function used within block neural autoregressive
            networks. Note this should be bijective and in some use cases should map real -> real.
        inverter (``Callable``, default ``None``): Callable that implements the required numerical method to invert the
            ``BlockAutoregressiveNetwork`` bijection. Must have the signature
            ``inverter(bijection, y, condition=None)``. Defaults to using a bisection
            search via ``AutoregressiveBisectionInverter``.
        permutation (``list[int]``, default ``None``): Permutation of the features, if
            ``None`` it will be randomly shuffled.
        random_seed (``int``, default ``0``): random seed

    Returns:
        ``Transformed``:
        _description_
    """
    key = jr.key(random_seed)
    base_dist = Normal(jnp.zeros(dim))
    activation = _get_activation(activation)
    if permutation is not None:
        permutation = jnp.array(permutation)

    def make_layer(key):  # bnaf layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = BlockAutoregressiveNetwork(
            bij_key,
            dim=base_dist.shape[-1],
            cond_dim=cond_dim,
            depth=nn_depth,
            block_dim=nn_block_dim,
            activation=activation,
            inverter=inverter,
        )
        return _add_default_permute(bijection, base_dist.shape[-1], perm_key, permutation)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers))
    return Transformed(base_dist, bijection)


@serialise_wrapper
def planar_flow(
    dim: int,
    cond_dim: int = None,
    flow_layers: int = 8,
    negative_slope: float = None,
    permutation: jnp.array = None,
    random_seed: int = 0,
    **mlp_kwargs,
) -> Transformed:
    """Planar flow as introduced in https://arxiv.org/pdf/1505.05770.pdf.

    This alternates between :class:`~flowjax.bijections.planar.Planar` layers and
    permutations. Note the definition here is inverted compared to the original paper.

    Args:
        key: Jax key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        cond_dim: Dimension of conditioning variables. Defaults to None.
        flow_layers: Number of flow layers. Defaults to 8.
        invert: Whether to invert the bijection. Broadly, True will prioritise a faster
            `inverse` methods, leading to faster `log_prob`, False will prioritise
            faster `transform` methods, leading to faster `sample`. Defaults to True.
        negative_slope: A positive float. If provided, then a leaky relu activation
            (with the corresponding negative slope) is used instead of tanh. This also
            provides the advantage that the bijection can be inverted analytically.
        **mlp_kwargs: Keyword arguments (excluding in_size and out_size) passed to
            the MLP (equinox.nn.MLP). Ignored when cond_dim is None.
    """
    key = jr.key(random_seed)
    base_dist = Normal(jnp.zeros(dim))

    def make_layer(key):  # Planar layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = Planar(
            bij_key,
            dim=base_dist.shape[-1],
            cond_dim=cond_dim,
            negative_slope=negative_slope,
            **mlp_kwargs,
        )
        return _add_default_permute(bijection, base_dist.shape[-1], perm_key, permutation)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers))
    return Transformed(base_dist, bijection)


@serialise_wrapper
def triangular_spline_flow(
    dim: int,
    cond_dim: int = None,
    flow_layers: int = 8,
    knots: int = 8,
    tanh_max_val: float = 3.0,
    random_seed: int = 0,
) -> Transformed:
    """Triangular spline flow.

    A single layer consists where each layer consists of a triangular affine
    transformation with weight normalisation, and an elementwise rational quadratic
    spline. Tanh is used to constrain to the input to [-1, 1] before spline
    transformations.

    Args:
        key: Jax random key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        cond_dim: The number of conditioning features. Defaults to None.
        flow_layers: Number of flow layers. Defaults to 8.
        knots: Number of knots in the splines. Defaults to 8.
        tanh_max_val: Maximum absolute value beyond which we use linear "tails" in the
            tanh function. Defaults to 3.0.
        invert: Whether to invert the bijection before transforming the base
            distribution. Defaults to True.
        init: Initialisation method for the lower triangular weights.
            Defaults to glorot_uniform().
    """
    key = jr.key(random_seed)
    base_dist = Normal(jnp.zeros(dim))
    init = glorot_uniform()

    def get_splines():
        fn = partial(RationalQuadraticSpline, knots=knots, interval=1)
        spline = eqx.filter_vmap(fn, axis_size=dim)()
        return Vmap(spline, in_axes=eqx.if_array(0))

    def make_layer(key):
        lt_key, perm_key, cond_key = jr.split(key, 3)
        weights = init(lt_key, (dim, dim))
        lt_weights = weights.at[jnp.diag_indices(dim)].set(1)
        tri_aff = TriangularAffine(jnp.zeros(dim), lt_weights)
        tri_aff = eqx.tree_at(
            lambda t: t.triangular, tri_aff, replace_fn=WeightNormalization
        )
        bijections = [
            LeakyTanh(tanh_max_val, (dim,)),
            get_splines(),
            Invert(LeakyTanh(tanh_max_val, (dim,))),
            tri_aff,
        ]

        if cond_dim is not None:
            linear_condition = AdditiveCondition(
                Linear(cond_dim, dim, use_bias=False, key=cond_key),
                (dim,),
                (cond_dim,),
            )
            bijections.append(linear_condition)

        bijection = Chain(bijections)
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers))
    return Transformed(base_dist, bijection)


_flow_registry = {
    "masked_autoregressive_flow": masked_autoregressive_flow,
    "coupling_flow": coupling_flow,
    "block_neural_autoregressive_flow": block_neural_autoregressive_flow,
    "planar_flow": planar_flow,
    "triangular_spline_flow": triangular_spline_flow,
}


def get_flow(flow: str) -> FlowLikelihood:
    """retreive flow"""
    return _flow_registry[flow]


def available_flows() -> list[str]:
    """Retreive available flows"""
    return list(_flow_registry.keys())


class FlowRegistrationError(Exception):
    """Flow Registration Error"""


def register_flow(func: Callable) -> Callable:
    """
    Register a custom flow

    Example:

    .. code::

        @nabu.register_flow
        def my_flow(...):
            ...
            return pytree

        assert "my_flow" in nabu.available_flows()

    """
    assert callable(func), "Invalid input, function needs to be callable."
    if func.__name__ in _flow_registry:
        warnings.warn(
            f"{func.__name__} is already registered. "
            "This action will overwrite the previous implementation."
        )
    registered_function = serialise_wrapper(func)

    @wraps(registered_function)
    def wrapper(*args, **kwargs):
        assert all(
            not callable(f) or isinstance(f, BijectorWrapper)
            for f in list(args) + list(kwargs.values())
        ), "Callable functions for the inputs are currently not supported"
        return registered_function(*args, **kwargs)

    _flow_registry.update({func.__name__: wrapper})
    return wrapper
