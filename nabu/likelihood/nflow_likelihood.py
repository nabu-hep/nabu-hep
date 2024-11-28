from jax import vmap
from flowjax.distributions import Transformed

from nabu.transform_base import PosteriorTransform

from .likelihood_base import LikelihoodBase


class FlowLikelihood(LikelihoodBase):
    model_type: str = "flow"

    __slots__ = ["_metadata"]

    def __init__(
        self,
        model: Transformed,
        metadata: dict,
        posterior_transform: PosteriorTransform = PosteriorTransform(),
    ):
        self._metadata = metadata

        super().__init__(
            model=model,
            posterior_transform=posterior_transform,
        )

    __dict__ = LikelihoodBase.__dict__

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
