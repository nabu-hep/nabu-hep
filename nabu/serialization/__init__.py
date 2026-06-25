# Copyright (c) 2026 Danny van Dyk

"""Typed, validated (de)serialisation of the ``.nabu`` header.

This package models the JSON header of a ``.nabu`` file as a tree of validated
:func:`dataclasses.dataclass` objects. :mod:`nabu.serialization.base` provides
the generic machinery (:class:`Deserializable`, :class:`Tagged`, and the error
hierarchy); the concrete likelihood/model/transform specifications are added in
:mod:`nabu.serialization.specs`.
"""

from .base import (
    Deserializable,
    SchemaError,
    SerializationError,
    Tagged,
    UnknownTagError,
)

__all__ = [
    "Deserializable",
    "Tagged",
    "SerializationError",
    "SchemaError",
    "UnknownTagError",
]


def __dir__():
    return __all__
