import warnings
from collections.abc import Callable, Generator, Sequence

import numpy as np
from scipy.stats import chi2, kstwo, kstwobign, norm

__all__ = ["Histogram", "weighted_kstest"]


def __dir__():
    return __all__


def weighted_kstest(
    vals: np.ndarray,
    cdf: Callable[[np.ndarray], np.ndarray],
    weights: np.ndarray = None,
) -> tuple[float, float, float]:
    r"""
    One-sample Kolmogorov-Smirnov test against ``cdf`` for (optionally) weighted data.

    The test statistic is the supremum distance between the weighted empirical CDF
    and ``cdf``. The p-value is computed from the Kish effective sample size
    :math:`n_{\rm eff} = (\sum w)^2 / \sum w^2` rather than the raw number of rows:

    * when all weights are equal (the unweighted case), ``n_eff`` is the integer
      sample size and the **exact** Kolmogorov-Smirnov distribution is used, exactly
      reproducing :func:`scipy.stats.kstest`;
    * with non-uniform weights ``n_eff`` is generally fractional, for which no exact
      finite-sample distribution exists, so the **asymptotic** Kolmogorov distribution
      is used instead.

    This is the appropriate test for importance samples with non-uniform weights:
    feeding a weighted-bootstrap (resampled, hence duplicated) set into an unweighted
    KS test violates the i.i.d. assumption and over-powers the test, because each
    duplicated point is counted as an independent draw.

    Args:
        vals (``np.ndarray``): sample values, shape ``(N,)``.
        cdf (``Callable``): cumulative distribution function to test against. Must be
            vectorised over a 1d array of sorted values.
        weights (``np.ndarray``, default ``None``): per-sample weights. If ``None``,
            taken as ``1`` (recovering the unweighted test).

    Returns:
        ``tuple[float, float, float]``:
        the KS statistic ``D``, the p-value, and the effective sample size.
    """
    vals = np.asarray(vals, dtype=float)
    weights = np.ones_like(vals) if weights is None else np.asarray(weights, dtype=float)

    if len(vals) != len(weights):
        raise ValueError(
            f"vals and weights must have the same length, got {len(vals)} and "
            f"{len(weights)}"
        )
    if len(vals) == 0:
        raise ValueError("vals must contain at least one sample")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    sumw = weights.sum()
    if not sumw > 0.0:
        raise ValueError("the total weight must be positive")

    order = np.argsort(vals, kind="mergesort")
    v, w = vals[order], weights[order]
    ecdf_hi = np.cumsum(w) / sumw  # weighted ECDF just after each point
    ecdf_lo = ecdf_hi - w / sumw  # weighted ECDF just before each point
    f = cdf(v)
    d = float(max(np.max(ecdf_hi - f), np.max(f - ecdf_lo)))
    n_eff = float(sumw**2 / np.sum(w**2))  # Kish effective sample size

    if np.allclose(w, w[0]):
        # Equally weighted points: the effective N is the integer sample size and the
        # exact KS distribution applies (matching scipy.stats.kstest).
        pvalue = float(kstwo.sf(d, len(w)))
    else:
        # Fractional effective N: fall back to the asymptotic Kolmogorov distribution.
        pvalue = float(kstwobign.sf(np.sqrt(n_eff) * d))
    return d, pvalue, n_eff


def sqrt_method(values, *args, **kwargs):
    return values - np.sqrt(values), values + np.sqrt(values)


def poisson_interval(
    sumw: np.ndarray,
    sumw2: np.ndarray,
    coverage: float = norm.cdf(1) - norm.cdf(-1),  # 0.6826894921370859 -> 1sigma
):
    """
    Frequentist coverage interval for Poisson-distributed observations

    Calculates the so-called 'Garwood' interval,
    c.f. https://www.ine.pt/revstat/pdf/rs120203.pdf or
    http://ms.mcmaster.ca/peter/s743/poissonalpha.html
    For weighted data, this approximates the observed count by ``sumw**2/sumw2``, which
    effectively scales the unweighted poisson interval by the average weight.
    This may not be the optimal solution: see https://arxiv.org/pdf/1309.1287.pdf for a
    proper treatment. When a bin is zero, the scale of the nearest nonzero bin is
    substituted to scale the nominal upper bound.
    If all bins zero, a warning is generated and interval is set to ``sumw``.
    Taken from Coffea

    Args:
        sumw (``np.ndarray``): Sum of weights vector
        sumw2 (``np.ndarray``): Sum weights squared vector
        coverage (``float``, default ``norm.cdf(1)-norm.cdf(-1)``):
            Central coverage interval, defaults to 68%
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
    r"""
    Create a histogram object for the underlying :math:`\chi^2` distribution.

    .. note::

        This class assumes that the base distribution of the flow is unit gaussian.

    Args:
        dim (``int``): number of features.
        bins (``Union[int, np.ndarray]``): If integer, indicates number of bins, if array, indicates
            bin edges.
        vals (``np.ndarray``): Sum of the feature values that deviate from the
            central Gaussian distriburion
        max_val (``float``, default ``None``): Maximum value that histogram can take.
            will only be used if ``bins`` input is ``int``.
        weights (``np.ndarray``, default ``None``): weight per value. If ``None`` taken as ``1``.
    """

    __slots__ = [
        "dim",
        "vals",
        "weights",
        "max_val",
        "bins",
        "sumw",
        "sumw2",
        "values",
        "variances",
        "bin_weights",
        "bin_width",
        "_kstest",
    ]

    def __init__(
        self,
        dim: int,
        bins: int,
        vals: np.ndarray,
        max_val: float = None,
        weights: np.ndarray = None,
    ) -> None:
        self.dim = dim
        self.vals = vals
        self.weights = (
            np.ones(len(self.vals)) if weights is None else np.asarray(weights, float)
        )
        if len(self.vals) != len(self.weights):
            raise ValueError(
                f"vals and weights must have the same length, got {len(self.vals)} "
                f"and {len(self.weights)}"
            )
        if np.any(self.weights < 0.0):
            raise ValueError("weights must be non-negative")
        if not np.sum(self.weights) > 0.0:
            raise ValueError("the total weight must be positive")

        if isinstance(bins, int):
            assert max_val is not None, "If bins are not defined, max_val is needed"
            self.max_val = max_val
            self.bins = np.linspace(0, max_val, bins + 1)
        else:
            self.bins = np.array(bins)
            self.max_val = max(bins)
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
        self._kstest = None

    @property
    def nbins(self) -> int:
        """Number of bins"""
        return len(self.bins) - 1

    @property
    def bin_mask(self) -> Generator[np.ndarray]:
        """Mask the values for each bin"""
        for left, right in self.bin_edges:
            yield (self.vals >= left) * (self.vals < right)

    @property
    def bin_edges(self) -> Generator[np.ndarray]:
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
        with warnings.catch_warnings(record=True):
            return (expected - self.values) / np.sqrt(self.variances)

    @property
    def yerr(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute y-error"""
        method = (
            poisson_interval
            if np.allclose(self.variances, np.around(self.variances))
            else sqrt_method
        )
        return calculate_relative(method, self.values, self.variances)

    @property
    def yerr_density(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute y-error for density distribution"""
        with warnings.catch_warnings(record=True):
            return self.density * self.yerr / self.values

    @property
    def xerr(self) -> tuple[np.ndarray, np.ndarray]:
        """
        compute errors on the x-axis

        Returns:
            ``Tuple[np.ndarray, np.ndarray]``:
            low and high errors
        """
        xerr = np.array(
            [
                [center - left, right - center]
                for (left, right), center in zip(self.bin_edges, self.bin_centers)
            ]
        )
        return xerr[:, 0], xerr[:, 1]

    @property
    def kstest_pval(self) -> float:
        """Weighted Kolmogorov-Smirnov p-value vs chi2(dim), using effective N"""
        if self._kstest is None:
            self._kstest = weighted_kstest(
                self.vals, lambda x: chi2.cdf(x, df=self.dim), self.weights
            )
        return self._kstest[1]

    @property
    def kstest_dval(self) -> float:
        """Weighted Kolmogorov-Smirnov statistic D vs chi2(dim)"""
        if self._kstest is None:
            self._kstest = weighted_kstest(
                self.vals, lambda x: chi2.cdf(x, df=self.dim), self.weights
            )
        return self._kstest[0]

    @property
    def effective_nobs(self) -> float:
        """Kish effective sample size of the (weighted) values"""
        if self._kstest is None:
            self._kstest = weighted_kstest(
                self.vals, lambda x: chi2.cdf(x, df=self.dim), self.weights
            )
        return self._kstest[2]

    @property
    def residuals_pvalue(self) -> float:
        """Compute the p-value for residuals"""
        pull = self.pull[:-1]  # K-1 independent variables
        return 1.0 - chi2.cdf(np.sum(pull**2), df=len(pull))

    def pull_mask(self, condition: Callable[[np.ndarray], Sequence[bool]]) -> np.ndarray:
        """Create a sample mask from the statistical pull"""

        sample_mask = []
        for pull_mask, bin_mask in zip(condition(self.pull), self.bin_mask):
            if pull_mask:
                sample_mask.append(bin_mask)

        return sum(sample_mask).astype(bool)
