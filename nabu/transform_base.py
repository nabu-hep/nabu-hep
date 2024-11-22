from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from .dalitz_utils import dalitz_to_square, square_to_dalitz

Array = Any


class PosteriorTransform:
    """Base class to handle transformations"""

    def __init__(self, name: Literal["mean_std", "dalitz"] = None, **kwargs):
        if name == "mean_std":
            mean, scale = jnp.array(kwargs["mean"]), jnp.array(kwargs["std"])

            def forward(x):
                return (x - mean) / scale

            def backward(x):
                return x * scale + mean

        elif name == "dalitz":
            md, ma, mb, mc = (
                kwargs["md"],
                kwargs["ma"],
                kwargs["mb"],
                kwargs["mc"],
            )

            def forward(x):
                return dalitz_to_square(x, md, ma, mb, mc)

            def backward(x):
                return square_to_dalitz(x, md, ma, mb, mc)

        elif name is None:
            forward = lambda x: x
            backward = lambda x: x

        else:
            raise NotImplementedError(f"{name} currently not implemented")

        self._forward = forward
        self._backward = backward
        self._meta = {"name": name}
        self._meta.update({"kwargs": kwargs})

    def serialise(self) -> dict:
        """Serialise transformer"""
        return self._meta

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
        return jnp.array(self.forward(x))

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
