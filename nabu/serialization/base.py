# Copyright (c) 2026 Danny van Dyk

"""Typed, validated (de)serialisation core for the ``.nabu`` header.

This module provides the generic machinery on top of which the concrete
likelihood/model/transform *specifications* (see :mod:`nabu.serialization.specs`)
are built. It is deliberately free of any ``nabu`` domain knowledge: it knows how
to turn a tree of :func:`dataclasses.dataclass` objects into plain
JSON-compatible ``dict``/``list`` structures and back, with validation and clear,
catchable errors.

The design is inspired by ``eos.deserializable.Deserializable`` (a minimal
``from_dict``/``make`` factory) and extends it with two capabilities the
``.nabu`` format needs:

* **recursion** into nested :class:`Deserializable` fields, and
* **tagged-union dispatch** for the polymorphic blocks of the header, which use
  the single-key ``{tag: payload}`` idiom (e.g. ``{"masked_autoregressive_flow":
  {...}}``).

Nothing here touches the binary ``equinox`` leaf payload; it models the JSON
header only.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from typing import Any, ClassVar, Union, get_args, get_origin, get_type_hints

__all__ = [
    "SerializationError",
    "SchemaError",
    "UnknownTagError",
    "Deserializable",
    "Tagged",
]


def __dir__():
    return __all__


class SerializationError(Exception):
    """Base class for all (de)serialisation failures of a ``.nabu`` header.

    Catching this type catches every structured error raised by this module,
    in contrast to the bare ``assert`` statements and opaque ``TypeError``/
    ``KeyError`` exceptions of the legacy loader.
    """


class SchemaError(SerializationError):
    """The data does not match the expected schema.

    Raised for unexpected/missing keys, wrong container shapes, failed
    construction, and malformed tagged unions.
    """


class UnknownTagError(SchemaError):
    """A discriminator tag is not registered in its union.

    Raised by :meth:`Tagged.from_tagged` when the tag (e.g. a flow, bijector or
    posterior-transform name) is not known. The error message lists the
    registered alternatives.
    """


# Resolved type hints are cached per class. ``get_type_hints`` evaluates the
# (string) annotations in the defining module's namespace, which is why this is
# robust to ``from __future__ import annotations`` in the spec modules.
_HINTS_CACHE: dict[type, dict[str, Any]] = {}


def _resolve_hints(cls: type) -> dict[str, Any]:
    """Return (and cache) the resolved type hints for a dataclass ``cls``."""
    cached = _HINTS_CACHE.get(cls)
    if cached is None:
        cached = get_type_hints(cls)
        _HINTS_CACHE[cls] = cached
    return cached


def _is_union(origin: Any) -> bool:
    """Whether ``origin`` denotes a typing union (``Union[...]`` or ``X | Y``)."""
    return origin is Union or origin is types.UnionType


def _coerce(hint: Any, value: Any) -> Any:
    """Convert a raw JSON ``value`` into the object described by ``hint``.

    Handles ``None``, optionals/unions, ``list``/``tuple`` containers, nested
    :class:`Deserializable` fields and :class:`Tagged` unions. Anything else
    (scalars, plain ``dict`` payloads, ``Any``) is returned unchanged.
    """
    # A JSON ``null`` is ``None`` regardless of the declared type.
    if value is None:
        return None

    origin = get_origin(hint)

    if _is_union(hint if origin is None else origin):
        members = [arg for arg in get_args(hint) if arg is not type(None)]
        if len(members) == 1:
            return _coerce(members[0], value)
        # General union: accept the first member that coerces cleanly.
        for member in members:
            try:
                return _coerce(member, value)
            except SerializationError:
                continue
        raise SchemaError(f"value {value!r} matches none of {members}")

    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise SchemaError(
                f"expected a sequence for {hint}, got {type(value).__name__}"
            )
        args = get_args(hint)
        if origin is tuple and args and args[-1] is not Ellipsis and len(args) > 1:
            # Fixed-length heterogeneous tuple: coerce positionally.
            if len(args) != len(value):
                raise SchemaError(
                    f"expected {len(args)} elements for {hint}, got {len(value)}"
                )
            return tuple(_coerce(arg, item) for arg, item in zip(args, value))
        elem_hint = args[0] if args else object
        items = [_coerce(elem_hint, item) for item in value]
        return tuple(items) if origin is tuple else items

    if isinstance(hint, type) and issubclass(hint, Deserializable):
        if issubclass(hint, Tagged):
            return hint.from_tagged(value)
        if not isinstance(value, Mapping):
            raise SchemaError(
                f"expected an object for {hint.__name__}, " f"got {type(value).__name__}"
            )
        return hint.from_dict(**value)

    return value


def _to_jsonable(value: Any) -> Any:
    """Convert a (possibly nested) spec object into JSON-compatible data.

    ``Tagged`` members serialise to their single-key ``{tag: payload}`` form;
    other :class:`Deserializable` objects to their field ``dict``. Numeric
    normalisation of NumPy types is intentionally left to the JSON encoder
    layer.
    """
    if isinstance(value, Tagged):
        return value.to_tagged()
    if isinstance(value, Deserializable):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


class Deserializable:
    """Mixin giving a :func:`dataclasses.dataclass` a validated dict round-trip.

    Subclasses must be dataclasses. The ``dict`` produced by :meth:`to_dict` is
    JSON-compatible (after the encoder layer normalises NumPy scalars), and
    :meth:`from_dict` reconstructs the object, recursing into nested
    ``Deserializable``/:class:`Tagged` fields and validating the input.
    """

    @classmethod
    def make(cls, **kwargs) -> Deserializable:
        """Construct ``cls`` from already-coerced keyword arguments.

        Construction errors (missing/extra/invalid fields, failed
        ``__post_init__`` validation) are re-raised as :class:`SchemaError` with
        context, mirroring ``eos.deserializable.Deserializable.make``.
        """
        if not (isinstance(cls, type) and issubclass(cls, Deserializable)):
            raise SchemaError(f"{cls!r} is not a Deserializable subclass")
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                f"cannot construct {cls.__name__} from fields {sorted(kwargs)}: {exc}"
            ) from exc

    @classmethod
    def from_dict(cls, **kwargs) -> Deserializable:
        """Build an instance from a JSON-decoded mapping of field values.

        Unknown keys and unresolvable values raise :class:`SchemaError` rather
        than being silently ignored.
        """
        if not dataclasses.is_dataclass(cls):
            raise SchemaError(f"{cls.__name__} is not a dataclass; cannot deserialise")

        valid = {field.name for field in dataclasses.fields(cls)}
        unknown = set(kwargs) - valid
        if unknown:
            raise SchemaError(
                f"{cls.__name__}: unexpected key(s) {sorted(unknown)}; "
                f"known fields: {sorted(valid)}"
            )

        hints = _resolve_hints(cls)
        coerced: dict[str, Any] = {}
        for name, raw in kwargs.items():
            try:
                coerced[name] = _coerce(hints.get(name, object), raw)
            except SerializationError as exc:
                raise SchemaError(f"{cls.__name__}.{name}: {exc}") from exc
        return cls.make(**coerced)

    def to_dict(self) -> dict:
        """Return a JSON-compatible ``dict`` of this object's fields."""
        if not dataclasses.is_dataclass(self):
            raise SchemaError(
                f"{type(self).__name__} is not a dataclass; cannot serialise"
            )
        return {
            field.name: _to_jsonable(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }


class Tagged(Deserializable):
    """A :class:`Deserializable` whose JSON form is the tagged ``{tag: {...}}``.

    Concrete members declare their discriminator via the class keyword
    ``tag=...``; the abstract root of a union declares ``root=True`` to own a
    fresh registry. This preserves the legacy on-disk shape (a single-key
    object) while replacing the fragile ``list(*d.items())`` unpacking and bare
    ``KeyError`` lookups with validated dispatch.

    Example::

        @dataclass(frozen=True)
        class BijectorSpec(Tagged, root=True):
            ...

        @dataclass(frozen=True)
        class AffineSpec(BijectorSpec, tag="Affine"):
            ...
    """

    #: Discriminator for a concrete member; ``None`` on abstract roots.
    tag: ClassVar[str | None] = None
    #: Maps tag -> concrete subclass. Each ``root=True`` class owns its own.
    _registry: ClassVar[dict[str, type]] = {}

    def __init_subclass__(cls, *, tag: str | None = None, root: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if root:
            cls._registry = {}
        if tag is not None:
            if not isinstance(tag, str):
                raise TypeError(f"tag for {cls.__name__} must be a str, got {tag!r}")
            existing = cls._registry.get(tag)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"tag {tag!r} is already registered to {existing.__name__}"
                )
            cls.tag = tag
            cls._registry[tag] = cls

    @classmethod
    def from_tagged(cls, data: Any) -> Tagged:
        """Dispatch a single-key ``{tag: payload}`` object to its member type."""
        if not isinstance(data, Mapping):
            raise SchemaError(
                f"{cls.__name__}: expected a single-key object, "
                f"got {type(data).__name__}"
            )
        if len(data) != 1:
            raise SchemaError(
                f"{cls.__name__}: expected exactly one tag, got keys {sorted(data)}"
            )
        ((tag, payload),) = data.items()
        target = cls._registry.get(tag)
        if target is None:
            raise UnknownTagError(
                f"{cls.__name__}: unknown tag {tag!r}; "
                f"registered tags: {sorted(cls._registry)}"
            )
        if not isinstance(payload, Mapping):
            raise SchemaError(
                f"{cls.__name__}: payload for tag {tag!r} must be an object, "
                f"got {type(payload).__name__}"
            )
        return target.from_dict(**payload)

    def to_tagged(self) -> dict:
        """Return the single-key ``{tag: fields}`` representation."""
        if self.tag is None:
            raise SchemaError(
                f"{type(self).__name__} has no tag; "
                "cannot serialise it as a tagged-union member"
            )
        return {self.tag: self.to_dict()}
