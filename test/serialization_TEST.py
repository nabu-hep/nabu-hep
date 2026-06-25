"""Unit tests for the (de)serialisation core in :mod:`nabu.serialization.base`.

These tests exercise the generic machinery only (validated dict round-trips,
nested fields, tagged-union dispatch and the error behaviour) using throwaway
dataclasses, so they are independent of the concrete ``nabu`` specs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import pytest

from nabu.serialization.base import (
    Deserializable,
    SchemaError,
    SerializationError,
    Tagged,
    UnknownTagError,
)


# ---------------------------------------------------------------------------
# Sample model: plain nested dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Point(Deserializable):
    x: float
    y: float = 0.0


@dataclass(frozen=True)
class Line(Deserializable):
    start: Point
    end: Point
    label: str | None = None


# ---------------------------------------------------------------------------
# Sample model: a tagged union (discriminated by ``tag``)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape(Tagged, root=True):
    """Abstract root of the shape union."""


@dataclass(frozen=True)
class Circle(Shape, tag="circle"):
    radius: float


@dataclass(frozen=True)
class Rectangle(Shape, tag="rectangle"):
    width: float
    height: float
    corners: tuple[int, ...] = ()


@dataclass(frozen=True)
class Drawing(Deserializable):
    """A dataclass embedding tagged unions and a list of plain dataclasses."""

    primary: Shape
    secondary: Shape | None = None
    points: list[Point] = field(default_factory=list)


# A second, independent union to check registries do not collide.
@dataclass(frozen=True)
class Animal(Tagged, root=True):
    pass


@dataclass(frozen=True)
class Dog(Animal, tag="dog"):
    name: str


# ---------------------------------------------------------------------------
# Plain dataclass round-trips
# ---------------------------------------------------------------------------


class TestPlainRoundTrip:
    def test_to_dict_includes_all_fields(self):
        assert Point(x=1.0, y=2.0).to_dict() == {"x": 1.0, "y": 2.0}

    def test_from_dict_uses_defaults(self):
        assert Point.from_dict(x=3.0) == Point(x=3.0, y=0.0)

    def test_nested_round_trip(self):
        line = Line(start=Point(1.0, 2.0), end=Point(3.0, 4.0), label="seg")
        data = line.to_dict()
        assert data == {
            "start": {"x": 1.0, "y": 2.0},
            "end": {"x": 3.0, "y": 4.0},
            "label": "seg",
        }
        assert Line.from_dict(**data) == line

    def test_optional_field_none_round_trips(self):
        line = Line(start=Point(1.0), end=Point(2.0))
        data = line.to_dict()
        assert data["label"] is None
        assert Line.from_dict(**data) == line


# ---------------------------------------------------------------------------
# Container coercion
# ---------------------------------------------------------------------------


class TestContainers:
    def test_list_of_dataclasses(self):
        drawing = Drawing(primary=Circle(1.0), points=[Point(1.0), Point(2.0, 3.0)])
        data = drawing.to_dict()
        assert data["points"] == [{"x": 1.0, "y": 0.0}, {"x": 2.0, "y": 3.0}]
        assert Drawing.from_dict(**data) == drawing

    def test_list_is_coerced_into_tuple_field(self):
        # JSON has no tuples: a list on disk must round-trip to a tuple field.
        rect = Rectangle.from_dict(width=1.0, height=2.0, corners=[1, 2, 3, 4])
        assert rect.corners == (1, 2, 3, 4)
        assert isinstance(rect.corners, tuple)

    def test_tuple_field_serialises_to_list(self):
        rect = Rectangle(width=1.0, height=2.0, corners=(1, 2))
        assert rect.to_dict()["corners"] == [1, 2]


# ---------------------------------------------------------------------------
# Tagged-union dispatch
# ---------------------------------------------------------------------------


class TestTaggedUnion:
    def test_to_tagged_shape(self):
        assert Circle(2.0).to_tagged() == {"circle": {"radius": 2.0}}

    def test_from_tagged_dispatches_to_member(self):
        shape = Shape.from_tagged({"rectangle": {"width": 1.0, "height": 2.0}})
        assert shape == Rectangle(width=1.0, height=2.0, corners=())

    def test_embedded_tagged_field_round_trips(self):
        drawing = Drawing(
            primary=Circle(1.0),
            secondary=Rectangle(2.0, 3.0, corners=(0, 1)),
        )
        data = drawing.to_dict()
        assert data["primary"] == {"circle": {"radius": 1.0}}
        assert data["secondary"] == {
            "rectangle": {"width": 2.0, "height": 3.0, "corners": [0, 1]}
        }
        assert Drawing.from_dict(**data) == drawing

    def test_optional_tagged_field_none(self):
        drawing = Drawing(primary=Circle(1.0))
        data = drawing.to_dict()
        assert data["secondary"] is None
        assert Drawing.from_dict(**data) == drawing

    def test_registries_are_isolated_per_root(self):
        assert set(Shape._registry) == {"circle", "rectangle"}
        assert set(Animal._registry) == {"dog"}
        assert Animal.from_tagged({"dog": {"name": "Rex"}}) == Dog(name="Rex")


# ---------------------------------------------------------------------------
# Error behaviour
# ---------------------------------------------------------------------------


class TestErrors:
    def test_error_hierarchy(self):
        assert issubclass(SchemaError, SerializationError)
        assert issubclass(UnknownTagError, SchemaError)

    def test_unexpected_key_is_rejected(self):
        with pytest.raises(SchemaError, match="unexpected key"):
            Point.from_dict(x=1.0, z=9.0)

    def test_missing_required_field_raises_schema_error(self):
        with pytest.raises(SchemaError, match="cannot construct Point"):
            Point.from_dict(y=1.0)

    def test_unknown_tag_lists_alternatives(self):
        with pytest.raises(UnknownTagError) as excinfo:
            Shape.from_tagged({"triangle": {}})
        message = str(excinfo.value)
        assert "triangle" in message
        assert "circle" in message and "rectangle" in message

    def test_tagged_requires_single_key(self):
        with pytest.raises(SchemaError, match="exactly one tag"):
            Shape.from_tagged({"circle": {"radius": 1.0}, "rectangle": {}})

    def test_tagged_requires_mapping(self):
        with pytest.raises(SchemaError, match="single-key object"):
            Shape.from_tagged(["circle"])

    def test_tagged_payload_must_be_object(self):
        with pytest.raises(SchemaError, match="must be an object"):
            Shape.from_tagged({"circle": 2.0})

    def test_nested_error_reports_field_path(self):
        # A bad value for the embedded tagged field surfaces the field name.
        with pytest.raises(SchemaError, match="Drawing.primary"):
            Drawing.from_dict(primary={"triangle": {}})

    def test_duplicate_tag_registration_is_rejected(self):
        with pytest.raises(ValueError, match="already registered"):

            @dataclass(frozen=True)
            class _AnotherCircle(Shape, tag="circle"):
                radius: float


# ---------------------------------------------------------------------------
# Sanity: the sample classes really are frozen dataclasses
# ---------------------------------------------------------------------------


def test_samples_are_dataclasses():
    for cls in (Point, Line, Circle, Rectangle, Drawing, Dog):
        assert dataclasses.is_dataclass(cls)
