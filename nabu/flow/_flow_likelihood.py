import jax.random as jr
import numpy as np
import optax
from flowjax.distributions import Transformed, StandardNormal
from jax import vmap

from nabu import Likelihood
from nabu.transform_base import PosteriorTransform

from ._train import fit, get_optimizer

__all__ = ["FlowLikelihood"]


def __dir__():
    return __all__


# pylint: disable=arguments-differ,too-many-arguments


class FlowLikelihood(Likelihood):
    model_type: str = "flow"

    __slots__ = ["_metadata"]

    def __init__(
        self,
        model: Transformed,
        metadata: dict,
        posterior_transform: PosteriorTransform = PosteriorTransform(),
    ):
        self._metadata = metadata
        assert isinstance(
            model.base_dist, StandardNormal
        ), "Only normal distribution as base distribution currently available."
        super().__init__(
            model=model,
            posterior_transform=posterior_transform,
        )

    def to_dict(self) -> dict:
        return self._metadata

    def inverse(self) -> Transformed:
        return vmap(self.model.bijection.inverse, in_axes=0)

    def __repr__(self) -> str:
        name = list(self._metadata.keys())[0]
        txt = name + "(\n"
        for key, item in self._metadata[name].items():
            txt += f"    {key}="
            if isinstance(item, dict):
                nm = list(item.keys())[0]
                txt += f"{nm}("
                for child_key, child_item in item[nm].items():
                    txt += f"{child_key} = {child_item}, "
                txt += ")"
            else:
                txt += f"{item}"
            txt += ",\n"
        return txt + ")"

    def fit_to_data(
        self,
        dataset: np.ndarray,
        L1_regularisation_coef: float = 0.0,
        L2_regularisation_coef: float = 0.0,
        condition: np.ndarray = None,
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        lr_scheduler: optax.Schedule = None,
        max_epochs: int = 100,
        max_patience: int = 5,
        check_every: int = 1,
        batch_size: int = 100,
        validation_probability: float = 0.1,
        verbose: bool = True,
        random_seed: int = np.random.randint(0, high=999999999999),
        plot_progress: str = None,
        metrics: list[callable] = None,
    ) -> dict[str, list[float]]:
        """
        Fit likelihood to the data

        Args:
            dataset (``np.ndarray``): _description_
            condition (``np.ndarray``, default ``None``): _description_
            learning_rate (``float``, default ``1e-4``): _description_
            optimizer (``str``, default ``"adam"``): _description_
            ls_scheduler (``optax.Schedule``, default ``None``): _description_
            max_epochs (``int``, default ``100``): _description_
            max_patience (``int``, default ``5``): _description_
            check_every (``int``, default ``1``): _description_
            batch_size (``int``, default ``100``): _description_
            validation_probability (``float``, default ``0.1``): _description_
            verbose (``bool``, default ``True``): _description_
            random_seed (``int``): _description_
            plot_progress (``str``, default ``None``): _description_

        Returns:
            ``dict[str, list[float]]``:
            Training history
        """
        optimizer = optax.inject_hyperparams(get_optimizer(optimizer))(
            learning_rate=learning_rate
        )
        flow, history = fit(
            key=jr.key(random_seed),
            dist=self.model,
            x=self.transform.backward(dataset),
            L1_regularisation_coef=L1_regularisation_coef,
            L2_regularisation_coef=L2_regularisation_coef,
            condition=condition,
            optimizer=optimizer,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
            val_prop=validation_probability,
            lr_scheduler=lr_scheduler,
            show_progress=verbose,
            check_every=check_every,
            plot_progress=plot_progress,
            metrics=metrics,
        )
        self._model = flow
        return history
