from typing import Text, Any

from flowjax.bijections import (
    Affine,
    Concatenate,
    Coupling,
    EmbedCondition,
    Exp,
    LeakyTanh,
    Loc,
    Planar,
    Power,
    RationalQuadraticSpline,
    Reshape,
    Scale,
    Scan,
    Sigmoid,
    SoftPlus,
    Stack,
    Tanh,
    TriangularAffine,
)

__all__ = [
    "get_bijector",
    "Affine",
    "Concatenate",
    "Coupling",
    "EmbedCondition",
    "Exp",
    "LeakyTanh",
    "Loc",
    "Planar",
    "Power",
    "RationalQuadraticSpline",
    "Reshape",
    "Scale",
    "Scan",
    "Sigmoid",
    "SoftPlus",
    "Stack",
    "Tanh",
    "TriangularAffine",
]


def __dir__():
    return __all__


BIJECTORS = {
    "Affine": Affine,
    "Concatenate": Concatenate,
    "Coupling": Coupling,
    "EmbedCondition": EmbedCondition,
    "Exp": Exp,
    "LeakyTanh": LeakyTanh,
    "Loc": Loc,
    "Power": Power,
    "Planar": Planar,
    "RationalQuadraticSpline": RationalQuadraticSpline,
    "Reshape": Reshape,
    "Scale": Scale,
    "Scan": Scan,
    "Sigmoid": Sigmoid,
    "SoftPlus": SoftPlus,
    "Stack": Stack,
    "Tanh": Tanh,
    "TriangularAffine": TriangularAffine,
}


def get_bijector(bijector: str):
    return BIJECTORS[bijector]
