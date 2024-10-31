import importlib
import inspect
import sys
from pathlib import Path
from typing import Callable, Text

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal, Transformed
from jax.nn import relu, sigmoid

from .maf import masked_autoregressive_flow


def build_flow(config_file: Text, model_file: Text, random_seed: int = 0) -> Transformed:
    """
    Build normalising flow model.

    Args:
        config_file (``Text``): model configuration file
        model_file (``Text``): saved model file
        random_seed (``int``, default ``0``): random seed

    Raises:
        ``FileNotFoundError``: if either configuration or model file does not exist

    Returns:
        ``Transformed``:
        normalising flow
    """

    config, model = Path(config_file), Path(model_file)
    if not config.is_file():
        raise FileNotFoundError(f"Cannot find configuration file at {config}")
    if not model.is_file():
        raise FileNotFoundError(f"Cannot find configuration file at {model}")

    with config.open(mode="r", encoding="utf-8") as f:
        flow_config = yaml.safe_load(f)

    transformer = {
        "affine": None,
        "rqs": RationalQuadraticSpline(**flow_config.get("transformer_kwargs", {})),
    }[flow_config.get("transformer", "affine")]

    activation = {"relu": relu, "sigmoid": sigmoid}[flow_config.get("activation", "relu")]

    key = jr.key(random_seed)
    flow = masked_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(flow_config["dim"])),
        transformer=transformer,
        nn_width=flow_config["nn_width"],
        nn_depth=flow_config["nn_depth"],
        flow_layers=flow_config["flow_layers"],
        invert=True,
        nn_activation=activation,
        permutation=list(reversed(range(flow_config["dim"]))),
    )

    flow = eqx.tree_deserialise_leaves(str(model), flow)

    return flow


def create_sampler(
    flow: Transformed,
    posterior_transform_file: Text = None,
    transformer_name: Text = "PosteriorTransform",
) -> Callable[[int], np.ndarray]:
    """
    Create sampler from normalising flow

    Args:
        flow (``Transformed``): normalising flow model
        posterior_transform_file (``Text``, default ``None``): file that defines
            posterior transformation
        transformer_name (``Text``, default ``"PosteriorTransform"``): name of the class
            or the function that performs transformation

    Raises:
        ``FileNotFoundError``: If ``posterior_transform_file`` file does not exist

    Returns:
        ``Callable[[int], np.ndarray]``:
        sampler function
    """

    if posterior_transform_file is None:
        posterior_transformer = (
            lambda x: x  # pylint: disable=unnecessary-lambda-assignment
        )
    else:
        posterior_transform_file = Path(posterior_transform_file)
        if not posterior_transform_file.is_file():
            raise FileNotFoundError(
                f"Cannot find configuration file at {posterior_transform_file}"
            )

        sys.path.append(str(posterior_transform_file.parent))
        post_transform = importlib.import_module(posterior_transform_file.stem)

        assert transformer_name in dir(
            post_transform
        ), f"{transformer_name} class/function is not in {posterior_transform_file}"

        if inspect.isclass(getattr(post_transform, transformer_name)):
            posterior_transformer = getattr(post_transform, transformer_name)()
        elif inspect.isfunction(getattr(post_transform, transformer_name)):
            posterior_transformer = getattr(post_transform, transformer_name)
        sys.path.pop(sys.path.index(str(posterior_transform_file.parent)))

    def sampler(size: int, random_seed: int = 0) -> np.ndarray:
        """
        Generate samples from the underlying normalising flow

        Args:
            size (``int``): number of samples to be generated
            random_seed (``int``, default ``0``): random seed

        Returns:
            ``np.ndarray``:
            generated samples
        """
        return np.array(posterior_transformer(flow.sample(jr.key(random_seed), (size,))))

    return sampler
