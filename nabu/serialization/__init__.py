# Copyright (c) 2026 Danny van Dyk

"""Typed, validated (de)serialisation of the ``.nabu`` header.

This package models the JSON header of a ``.nabu`` file as a tree of validated
:func:`dataclasses.dataclass` objects. :mod:`nabu.serialization.base` provides
the generic machinery (:class:`Deserializable`, :class:`Tagged`, and the error
hierarchy); :mod:`nabu.serialization.specs` adds the concrete
likelihood/model/transform specifications.
"""

from .base import (
    Deserializable,
    SchemaError,
    SerializationError,
    Tagged,
    UnknownTagError,
)
from .specs import (
    FORMAT_VERSION,
    BijectorSpec,
    BlockNeuralAutoregressiveFlowSpec,
    CouplingFlowSpec,
    DalitzTransformSpec,
    FlowSpec,
    GenericFlowSpec,
    IdentityTransformSpec,
    LikelihoodSpec,
    MaskedAutoregressiveFlowSpec,
    PlanarFlowSpec,
    PosteriorTransformSpec,
    ShiftScaleTransformSpec,
    TriangularSplineFlowSpec,
    UserDefinedTransformSpec,
)

__all__ = [
    # base machinery
    "Deserializable",
    "Tagged",
    "SerializationError",
    "SchemaError",
    "UnknownTagError",
    # header schema
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
