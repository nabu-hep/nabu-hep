# Copyright (c) 2026 Danny van Dyk

"""Concrete specifications for the ``.nabu`` header.

This module builds the domain-specific layer on top of
:mod:`nabu.serialization.base`. Each class is a validated
:func:`dataclasses.dataclass` that mirrors one block of the JSON header
documented in ``doc/file-format.rst``:

* :class:`LikelihoodSpec` -- the top-level header.
* :class:`FlowSpec` and its concrete subclasses -- the ``model`` block. The five
  built-in flows each get a typed dataclass; flows added via
  :func:`nabu.flow.register_flow` round-trip through :class:`GenericFlowSpec`.
* :class:`BijectorSpec` -- a nested ``transformer`` bijection.
* :class:`PosteriorTransformSpec` and its subclasses -- the
  ``posterior_transform`` block.

Every spec carries a ``build`` method that reconstructs the corresponding
*untrained* runtime object (a :class:`~nabu.Likelihood`, a ``flowjax`` bijection,
or a :class:`~nabu.transform_base.PosteriorTransform`). The registries
(``nabu.flow``) and runtime classes are imported lazily inside those methods so
that importing this module stays cheap and free of import cycles.

The on-disk byte format is unchanged: the polymorphic blocks keep the single-key
``{tag: payload}`` shape, and an identity posterior transform is still the empty
object ``{}``. The only addition is the top-level ``format_version`` field, which
defaults to :data:`FORMAT_VERSION` when absent (i.e. for files written before it
existed).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .base import (
    Deserializable,
    SchemaError,
    SerializationError,
    Tagged,
    UnknownTagError,
)

__all__ = [
    "FORMAT_VERSION",
    "LikelihoodSpec",
    "FlowSpec",
    "MaskedAutoregressiveFlowSpec",
    "CouplingFlowSpec",
    "BlockNeuralAutoregressiveFlowSpec",
    "PlanarFlowSpec",
    "TriangularSplineFlowSpec",
    "GenericFlowSpec",
    "BijectorSpec",
    "PosteriorTransformSpec",
    "IdentityTransformSpec",
    "ShiftScaleTransformSpec",
    "DalitzTransformSpec",
    "UserDefinedTransformSpec",
]


def __dir__():
    return __all__


#: The highest ``.nabu`` header schema version this module can read/write.
#: Files lacking the field are treated as version ``1``.
FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Bijectors (nested ``transformer`` bijections)
# ---------------------------------------------------------------------------


def _configure_bijector(name: str, kwargs: Mapping) -> Any:
    """Return a configured ``BijectorWrapper`` ready to be passed to a flow.

    List-valued arguments are converted to tuples, because several ``flowjax``
    bijections expect tuple-valued shape arguments. This centralises the
    list-to-tuple coercion that the legacy loader performed inline.
    """
    from nabu.flow import get_bijector

    configured = {
        key: (tuple(value) if isinstance(value, list) else value)
        for key, value in kwargs.items()
    }
    return get_bijector(name)(**configured)


@dataclass(frozen=True)
class BijectorSpec(Tagged, root=True):
    """A nested bijection, serialised as ``{name: kwargs}``.

    Bijector keyword arguments mirror the (volatile, ``flowjax``-owned)
    constructor signatures, so they are kept as an opaque ``kwargs`` mapping
    rather than a dataclass per bijector. The discriminating ``name`` is
    validated against :func:`nabu.flow.available_bijectors` at parse time.
    """

    name: str
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_tagged(cls, data: Any) -> BijectorSpec:
        if not isinstance(data, Mapping):
            raise SchemaError(
                f"BijectorSpec: expected a single-key object, "
                f"got {type(data).__name__}"
            )
        if len(data) != 1:
            raise SchemaError(
                f"BijectorSpec: expected exactly one bijector name, "
                f"got keys {sorted(data)}"
            )
        ((name, kwargs),) = data.items()
        if not isinstance(kwargs, Mapping):
            raise SchemaError(
                f"BijectorSpec: arguments for {name!r} must be an object, "
                f"got {type(kwargs).__name__}"
            )
        from nabu.flow import available_bijectors

        if name not in available_bijectors():
            raise UnknownTagError(
                f"BijectorSpec: unknown bijector {name!r}; "
                f"available: {sorted(available_bijectors())}"
            )
        return cls(name=name, kwargs=dict(kwargs))

    def to_tagged(self) -> dict:
        return {self.name: dict(self.kwargs)}

    def build(self) -> Any:
        """Return the configured ``flowjax`` bijection wrapper."""
        return _configure_bijector(self.name, self.kwargs)


# ---------------------------------------------------------------------------
# Flows (the ``model`` block)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowSpec(Tagged, root=True):
    """Abstract root for the ``model`` block (``model_type == 'flow'``).

    Concrete subclasses mirror the constructor signature of the corresponding
    flow in :mod:`nabu.flow`. Flows registered at runtime via
    :func:`nabu.flow.register_flow` -- which therefore have no dedicated
    dataclass -- are represented by :class:`GenericFlowSpec`.
    """

    @classmethod
    def from_tagged(cls, data: Any) -> FlowSpec:
        if not isinstance(data, Mapping):
            raise SchemaError(
                f"FlowSpec: expected a single-key object, got {type(data).__name__}"
            )
        if len(data) != 1:
            raise SchemaError(
                f"FlowSpec: expected exactly one flow name, got keys {sorted(data)}"
            )
        ((tag, payload),) = data.items()
        if not isinstance(payload, Mapping):
            raise SchemaError(
                f"FlowSpec: arguments for {tag!r} must be an object, "
                f"got {type(payload).__name__}"
            )
        target = cls._registry.get(tag)
        if target is not None:
            return target.from_dict(**payload)

        # Fall back to a generic spec for custom (register_flow) flows.
        from nabu.flow import available_flows

        if tag in available_flows():
            return GenericFlowSpec(name=tag, kwargs=dict(payload))
        raise UnknownTagError(
            f"FlowSpec: unknown flow {tag!r}; built-in: {sorted(cls._registry)}; "
            f"registered: {sorted(available_flows())}"
        )

    def build(self, random_seed: int) -> Any:
        """Reconstruct the untrained :class:`~nabu.Likelihood` skeleton."""
        from nabu.flow import get_flow

        kwargs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        transformer = kwargs.get("transformer")
        if isinstance(transformer, BijectorSpec):
            kwargs["transformer"] = transformer.build()
        kwargs["random_seed"] = random_seed
        return get_flow(self.tag)(**kwargs)


@dataclass(frozen=True)
class MaskedAutoregressiveFlowSpec(FlowSpec, tag="masked_autoregressive_flow"):
    """Spec for :func:`nabu.flow.masked_autoregressive_flow`."""

    dim: int
    transformer: BijectorSpec | None = None
    cond_dim: int | None = None
    flow_layers: int = 8
    nn_width: int = 50
    nn_depth: int = 1
    activation: str = "relu"
    permutation: str = "reversed"
    random_seed: int = 0


@dataclass(frozen=True)
class CouplingFlowSpec(FlowSpec, tag="coupling_flow"):
    """Spec for :func:`nabu.flow.coupling_flow`."""

    dim: int
    transformer: BijectorSpec | None = None
    cond_dim: int | None = None
    flow_layers: int = 8
    nn_width: int | list[int] = 50
    activation: str = "relu"
    permutation: str = "reversed"
    random_seed: int = 0


@dataclass(frozen=True)
class BlockNeuralAutoregressiveFlowSpec(FlowSpec, tag="block_neural_autoregressive_flow"):
    """Spec for :func:`nabu.flow.block_neural_autoregressive_flow`.

    ``inverter`` is a callable in the constructor and is not serialisable; only
    its absence (``None``, the default bisection inverter) round-trips.
    """

    dim: int
    cond_dim: int | None = None
    nn_depth: int = 1
    nn_block_dim: int = 8
    flow_layers: int = 1
    activation: str = "sigmoid"
    inverter: str | None = None
    permutation: str = "reversed"
    random_seed: int = 0


@dataclass(frozen=True)
class PlanarFlowSpec(FlowSpec, tag="planar_flow"):
    """Spec for :func:`nabu.flow.planar_flow`.

    The constructor accepts ``**mlp_kwargs`` for the conditioner MLP; these are
    held in :attr:`mlp_kwargs` and splatted back at build time.
    """

    dim: int
    cond_dim: int | None = None
    flow_layers: int = 8
    negative_slope: float | None = None
    permutation: str = "reversed"
    random_seed: int = 0
    mlp_kwargs: dict = field(default_factory=dict)

    def build(self, random_seed: int) -> Any:
        from nabu.flow import get_flow

        kwargs = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "mlp_kwargs"
        }
        kwargs["random_seed"] = random_seed
        return get_flow("planar_flow")(**kwargs, **self.mlp_kwargs)


@dataclass(frozen=True)
class TriangularSplineFlowSpec(FlowSpec, tag="triangular_spline_flow"):
    """Spec for :func:`nabu.flow.triangular_spline_flow`."""

    dim: int
    cond_dim: int | None = None
    flow_layers: int = 8
    knots: int = 8
    tanh_max_val: float = 3.0
    permutation: str = "reversed"
    random_seed: int = 0


@dataclass(frozen=True)
class GenericFlowSpec(FlowSpec):
    """Fallback spec for custom flows without a dedicated dataclass.

    Holds the flow name and an opaque argument mapping, preserving the
    ``{name: kwargs}`` round-trip for flows added via
    :func:`nabu.flow.register_flow`.
    """

    name: str
    kwargs: dict = field(default_factory=dict)

    def to_tagged(self) -> dict:
        return {self.name: dict(self.kwargs)}

    def build(self, random_seed: int) -> Any:
        from nabu.flow import get_flow

        kwargs = dict(self.kwargs)
        kwargs["random_seed"] = random_seed
        transformer = kwargs.get("transformer")
        if transformer is not None:
            # Validate the transformer's shape (single tagged bijection) through
            # BijectorSpec, so a malformed header raises a structured
            # SchemaError/UnknownTagError rather than a raw ValueError/TypeError.
            kwargs["transformer"] = BijectorSpec.from_tagged(transformer).build()
        return get_flow(self.name)(**kwargs)


# ---------------------------------------------------------------------------
# Posterior transforms (the ``posterior_transform`` block)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorTransformSpec(Tagged, root=True):
    """Abstract root for the ``posterior_transform`` block.

    The empty object ``{}`` denotes the identity transform; every other variant
    uses the single-key ``{tag: payload}`` form.
    """

    @classmethod
    def from_tagged(cls, data: Any) -> PosteriorTransformSpec:
        if not isinstance(data, Mapping):
            raise SchemaError(
                f"PosteriorTransformSpec: expected an object, "
                f"got {type(data).__name__}"
            )
        if len(data) == 0:
            return IdentityTransformSpec()
        return super().from_tagged(data)

    def build(self) -> Any:
        """Reconstruct the runtime :class:`~nabu.transform_base.PosteriorTransform`."""
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityTransformSpec(PosteriorTransformSpec):
    """The identity transform, serialised as the empty object ``{}``."""

    def to_tagged(self) -> dict:
        return {}

    def build(self) -> Any:
        from nabu.transform_base import PosteriorTransform

        return PosteriorTransform()


@dataclass(frozen=True)
class ShiftScaleTransformSpec(PosteriorTransformSpec, tag="shift-scale"):
    """A per-feature shift/scale transform, with optional logarithmic axes."""

    shift: list[float]
    scale: list[float]
    log_axes: list[int] | None = None
    log_shift: list[float] | None = None
    log_scale: list[float] | None = None

    def build(self) -> Any:
        from nabu.transform_base import PosteriorTransform

        return PosteriorTransform(
            "shift-scale",
            shift=self.shift,
            scale=self.scale,
            log_axes=self.log_axes,
            log_shift=self.log_shift,
            log_scale=self.log_scale,
        )


@dataclass(frozen=True)
class DalitzTransformSpec(PosteriorTransformSpec, tag="dalitz"):
    """A Dalitz-coordinate transform."""

    md: float
    ma: float
    mb: float
    mc: float

    def build(self) -> Any:
        from nabu.transform_base import PosteriorTransform

        return PosteriorTransform(
            "dalitz", md=self.md, ma=self.ma, mb=self.mb, mc=self.mc
        )


@dataclass(frozen=True)
class UserDefinedTransformSpec(PosteriorTransformSpec):
    """A user-defined transform.

    Its ``forward``/``backward`` callables cannot be represented in the JSON
    header. Rather than silently degrading to the identity transform on load
    (as the legacy format did), serialising this spec raises immediately, so the
    loss is surfaced at save time. It is intentionally not registered for
    dispatch, since no on-disk form can produce it.
    """

    def to_tagged(self) -> dict:
        raise SerializationError(
            "user-defined posterior transforms cannot be serialised; "
            "save a shift-scale/dalitz transform instead, or re-apply the "
            "user-defined transform manually after loading"
        )

    def build(self) -> Any:
        raise SerializationError(
            "user-defined posterior transforms cannot be deserialised"
        )


# ---------------------------------------------------------------------------
# Top-level header
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LikelihoodSpec(Deserializable):
    """The complete ``.nabu`` JSON header.

    Field names and order mirror the on-disk header for byte-compatibility:
    ``model_type``, ``model``, ``posterior_transform`` and the (diagnostic)
    package ``version``, with the new ``format_version`` appended.
    """

    model_type: str
    model: FlowSpec
    posterior_transform: PosteriorTransformSpec
    version: str | None = None
    format_version: int = FORMAT_VERSION

    def __post_init__(self):
        if self.model_type != "flow":
            raise SchemaError(
                f"unsupported model_type {self.model_type!r}; "
                "only 'flow' is currently supported"
            )
        if self.format_version > FORMAT_VERSION:
            raise SchemaError(
                f"file format version {self.format_version} is newer than the "
                f"supported version {FORMAT_VERSION}; please update nabu"
            )

    def build_skeleton(self, random_seed: int) -> Any:
        """Build the untrained :class:`~nabu.Likelihood` (model + transform).

        The trained array leaves are *not* restored here; the caller is expected
        to read them into ``likelihood.model`` afterwards (see
        :meth:`nabu.Likelihood.load`).
        """
        likelihood = self.model.build(random_seed)
        likelihood.transform = self.posterior_transform.build()
        return likelihood
