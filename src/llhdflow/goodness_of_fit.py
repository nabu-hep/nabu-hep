import warnings
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import chi2, norm

__all__ = ["Histogram"]


def __dir__():
    return __all__


def sqrt_method(values, _):
    return values - np.sqrt(values), values + np.sqrt(values)


def poisson_interval(
    sumw: np.ndarray,
    sumw2: np.ndarray,
    coverage: float = norm.cdf(1) - norm.cdf(-1),  # 0.6826894921370859 -> 1sigma
):
    """Frequentist coverage interval for Poisson-distributed observations
    Parameters
    ----------
        sumw : numpy.ndarray
            Sum of weights vector
        sumw2 : numpy.ndarray
            Sum weights squared vector
        coverage : float, optional
            Central coverage interval, defaults to 68%
    Calculates the so-called 'Garwood' interval,
    c.f. https://www.ine.pt/revstat/pdf/rs120203.pdf or
    http://ms.mcmaster.ca/peter/s743/poissonalpha.html
    For weighted data, this approximates the observed count by ``sumw**2/sumw2``, which
    effectively scales the unweighted poisson interval by the average weight.
    This may not be the optimal solution: see https://arxiv.org/pdf/1309.1287.pdf for a
    proper treatment. When a bin is zero, the scale of the nearest nonzero bin is
    substituted to scale the nominal upper bound.
    If all bins zero, a warning is generated and interval is set to ``sumw``.
    # Taken from Coffea
    """
    scale = np.empty_like(sumw)
    scale[sumw != 0] = sumw2[sumw != 0] / sumw[sumw != 0]
    if np.sum(sumw == 0) > 0:
        missing = np.where(sumw == 0)
        available = np.nonzero(sumw)
        if len(available[0]) == 0:
            warnings.warn(
                "All sumw are zero!  Cannot compute meaningful error bars",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.vstack([sumw, sumw])
        nearest = np.sum(
            [np.subtract.outer(d, d0) ** 2 for d, d0 in zip(available, missing)]
        ).argmin(axis=0)
        argnearest = tuple(dim[nearest] for dim in available)
        scale[missing] = scale[argnearest]
    counts = sumw / scale
    lo = scale * chi2.ppf((1 - coverage) / 2, 2 * counts) / 2.0
    hi = scale * chi2.ppf((1 + coverage) / 2, 2 * (counts + 1)) / 2.0
    interval = np.array([lo, hi])
    interval[interval == np.nan] = 0.0  # chi2.ppf produces nan for counts=0
    return interval


def calculate_relative(method_fcn, values, variances):
    return np.abs(method_fcn(values, variances) - values)


class Histogram:
    """
    _summary_

    Args:
        dim (``int``): _description_
        bins (``Union[int, np.ndarray]``): _description_
        vals (``np.ndarray``): _description_
        max_val (``Optional[float]``, default ``None``): _description_
        weights (``np.ndarray``, default ``None``): _description_
    """

    def __init__(
        self,
        dim: int,
        bins: Union[int, np.ndarray],
        vals: np.ndarray,
        max_val: Optional[float] = None,
        weights: np.ndarray = None,
    ) -> None:
        self.dim = dim
        self.vals = vals
        self.weights = weights or np.ones(len(self.vals))
        assert len(self.vals) == len(self.weights), "Invalid shape"

        if isinstance(bins, int):
            assert max_val is not None, "If bins are not defined, max_val is needed"
            self.max_val = max_val
            self.bins = np.linspace(0, self.max_val, self.bins + 1)
        else:
            self.max_val = max(self.bins)
        self.bin_width = self.bins[1:] - self.bins[:-1]

        self.sumw = np.sum(self.weights)
        self.sumw2 = np.sum(self.weights**2)

        val, var = [], []
        for mask in self.bin_mask:
            w = self.weights[mask]
            val.append(w.sum())
            var.append(np.sum(w**2))
        self.values = np.array(val)
        self.variances = np.array(var)
        self.bin_weights = self.values / self.sumw

    @property
    def nbins(self) -> int:
        """Number of bins"""
        return len(self.bins) - 1

    @property
    def bin_mask(self):
        """Mask the values for each bin"""
        for left, right in self.bin_edges:
            yield (self.vals >= left) * (self.vals < right)

    @property
    def bin_edges(self):
        """Get bin edges"""
        for n in range(len(self.bins) - 1):
            yield self.bins[n : n + 2]

    @property
    def bin_centers(self) -> np.ndarray:
        """retreive bin centers"""
        return self.bins[:-1] + (self.bin_width / 2)

    @property
    def density(self) -> np.ndarray:
        """compute density"""
        total = self.values.sum() * self.bin_width
        return self.values / np.where(total > 0.0, total, 1.0)

    @property
    def pull(self) -> np.ndarray:
        """compute pull"""
        bin_prob = chi2.cdf(self.bins[1:], df=self.dim) - chi2.cdf(
            self.bins[:-1], df=self.dim
        )  # probability of getting events in each bin
        expected = bin_prob * self.values.sum()  # number of expected events in each bin
        # expected - observed / sqrt(var)
        return (expected - self.values) / np.sqrt(self.variances)

    @property
    def yerr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute y-error"""
        method = (
            poisson_interval
            if np.allclose(self.variances, np.around(self.variances))
            else sqrt_method
        )
        return calculate_relative(method, self.values, self.variances)

    @property
    def xerr(self):
        """compute x error"""
        los, his = [], []
        for (left, right), center in zip(self.bin_edges, self.bin_centers):
            los.append(center - left)
            his.append(right - center)
        return np.array(los), np.array(his)

    def pull_mask(self, condition: Callable[[np.ndarray], Sequence[bool]]) -> np.ndarray:
        """Create a sample mask from the statistical pull"""

        sample_mask = []
        for pull_mask, bin_mask in zip(condition(self.pull), self.bin_mask):
            if pull_mask:
                sample_mask.append(bin_mask)

        return sum(sample_mask).astype(bool)
