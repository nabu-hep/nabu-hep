from typing import Any
import jax.numpy as jnp
import numpy as np

Array = Any


class PosteriorTransformBase:
    """Base class to handle transformations"""

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
        return x

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
        return y

    def _forward(self, x: Array) -> jnp.ndarray:
        """
        Preloader for forward transformer.

        Args:
            x (``Array``): input data

        Returns:
            ``jnp.ndarray``:
            Transformed data
        """
        return jnp.array(self.forward(x))

    def _backward(self, y: jnp.array) -> np.ndarray:
        """
        Preloader for backward transformer.

        Args:
            y (``Array``): transformed data

        Returns:
            ``np.ndarray``:
            _description_
        """
        return np.array(self.backward(np.array(y)))
