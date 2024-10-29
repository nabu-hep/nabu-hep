import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, norm


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


@dataclass
class Histogram:
    dim: int
    bins: Union[int, np.ndarray]
    vals: np.ndarray
    max_val: Optional[float] = None
    weights: np.ndarray = None

    def __post_init__(self):
        if isinstance(self.bins, int):
            self.bins = np.linspace(0, self.max_val, self.bins + 1)
        self.bin_width = self.bins[1:] - self.bins[:-1]
        if self.weights is None:
            self.weights = np.ones(len(self.vals))
        assert len(self.vals) == len(self.weights), "Invalid shape"
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
        return len(self.bins) - 1

    @property
    def bin_mask(self):
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


def chi2_analysis(gaussian: np.ndarray, plot_name: str = None, **hist_kwargs) -> None:
    hist = Histogram(
        dim=gaussian.shape[1],
        bins=hist_kwargs.get("bins", 100),
        max_val=hist_kwargs.get("max_val", 20.0),
        vals=np.sum(gaussian**2, axis=1),
        weights=hist_kwargs.get("weights", None),
    )

    x = np.linspace(0, hist.max_val, 500)
    chi2p = chi2.pdf(x, df=hist.dim)

    fig = plt.figure()
    size = fig.get_size_inches()
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(size[0], size[1] * 1.25),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05, "wspace": 0.0},
    )

    errors = {"yerr": hist.density * hist.yerr / hist.values}
    # if len(np.unique(hist.bin_width)) != 1:
    #     errors.update({"xerr": hist.xerr})

    ax0.errorbar(
        hist.bin_centers,
        hist.density,
        **errors,
        fmt=".",
        lw=1,
        color="k",
        elinewidth=1,
        capsize=4,
    )
    ax0.set_yscale("log")

    # plt.xlim([15,21])
    ax0.set_xlim([-0.2, 20.2])
    ax0.set_ylim([5e-4, 0.2])
    ax0.plot(x, chi2p, color="tab:blue", label=rf"$\chi^2(\nu={hist.dim})$")

    ymin, ymax = ax0.get_ylim()
    ax0.set_ylim(ymin, ymax)
    for cl in [0.68, 0.95, 0.99]:
        p = chi2.isf(1.0 - cl, hist.dim)
        ax0.plot(
            [p] * 2,
            [ax0.get_ylim()[0], chi2.pdf(p, df=hist.dim)],
            color="tab:blue",
            linestyle="--",
        )
        ax0.text(
            p,
            ymin * 1.2,
            rf"${cl*100:.0f}\% " + r"{\rm\ CL}$",
            ha="right",
            va="bottom",
            rotation=90,
            fontsize=20,
        )
        ax1.axvline(p, color="gray", linestyle="--", zorder=0)

    color = np.array(["gray"] * hist.nbins, dtype=object)
    pull = hist.pull
    color[(abs(pull) > 1) & (abs(pull) <= 3)] = "gold"
    color[abs(pull) > 3] = "firebrick"
    ax1.bar(hist.bin_centers, hist.pull, width=hist.bin_width, color=color.tolist())
    ax1.set_ylim([-5, 5])

    if plot_name is not None:
        plt.savefig(plot_name)
    else:
        plt.show()
    plt.close()
