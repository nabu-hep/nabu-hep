import numpy as np
from flowjax.distributions import AbstractTransformed
from scipy.stats import chi2
from jax import vmap

from nabu.goodness_of_fit import Histogram

__all__ = ["GoodnessOfFit"]


def __dir__():
    return __all__


class GoodnessOfFit:
    """
    Compute goodness of fit metrics

    Args:
        prob_per_bin (``float``, default ``0.1``): probability per bin
    """

    def __init__(self, prob_per_bin: float = 0.1):
        self.prob_per_bin = prob_per_bin

    def __call__(
        self, dist: AbstractTransformed, test_data: np.ndarray
    ) -> dict[str, float]:
        dim = test_data.shape[-1]
        chi2_dist = np.sum(
            vmap(dist.bijection.inverse, in_axes=0)(test_data) ** 2, axis=1
        )
        bins = chi2.ppf(
            np.linspace(0.0, 1.0, int(np.ceil(1.0 / self.prob_per_bin)) + 1), df=dim
        )
        hist = Histogram(dim=dim, bins=bins, vals=chi2_dist)
        return {"kstest_pvalue": hist.kstest_pval, "chi2_pvalue": hist.residuals_pvalue}
