"""Fill summary plot"""
from collections.abc import Sequence

try:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif"})
except ImportError as err:
    raise NotImplementedError("Summary plot requires matplotlib") from err
import numpy as np
from scipy.stats import chi2

from .goodness_of_fit import Histogram
from .likelihood import Likelihood

# pylint: disable=too-many-arguments, too-many-locals


def summary_plot(
    likelihood: Likelihood,
    test_data: np.ndarray,
    weights: Sequence[float] = None,
    bins: Sequence[float] = None,
    hist_max_value: float = 20.0,
    prob_per_bin: float = None,
    add_xerr: bool = False,
    confidence_level: Sequence[float] = (0.68, 0.95, 0.99),
) -> tuple:
    """
    Prepare summary plot

    Args:
        likelihood (``Likelihood``): _description_
        test_data (``np.ndarray``): _description_
        weights (``Sequence[float]``, default ``None``): _description_
        bins (``Sequence[float]``, default ``None``): _description_
        hist_max_value (``float``, default ``20.0``): _description_
        prob_per_bin (``float``, default ``None``): _description_
        add_xerr (``bool``, default ``False``): _description_
        confidence_level (``Sequence[float]``, default ``(0.68, 0.95, 0.99)``): _description_

    Returns:
        Matplotlib figure and two axes
    """
    assert not all(
        x is None for x in [bins, prob_per_bin]
    ), "Both `prob_per_bin` and `bins` argument can not be `None`."

    # Developer option:
    if likelihood == "__dev__":
        deviations = test_data
    else:
        deviations = likelihood.compute_inverse(test_data)

    if prob_per_bin is not None:
        bins = np.hstack(
            [
                chi2.ppf(np.arange(0.0, 1, prob_per_bin), df=deviations.shape[1]),
                [hist_max_value],
            ]
        )

    chi2_test = np.sum(deviations**2, axis=1)

    hist = Histogram(
        dim=deviations.shape[1],
        bins=bins,
        max_val=hist_max_value,
        vals=chi2_test,
        weights=weights,
    )

    hist_pval_test = Histogram(
        dim=deviations.shape[1],
        bins=chi2.ppf(np.arange(0.0, 1.1, 0.1), df=deviations.shape[1]),
        vals=chi2_test,
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

    errors = {"yerr": hist.yerr_density}
    if len(np.unique(hist.bin_width)) != 1 and add_xerr:
        errors.update({"xerr": hist.xerr})

    ax0.errorbar(
        hist.bin_centers,
        hist.density,
        **errors,
        fmt=".",
        lw=1,
        color="k",
        elinewidth=1,
        capsize=4,
        label=r"${\rm Transformed}$" + "\n" + r"${\rm samples}$",
        zorder=100,
    )
    ax0.set_yscale("log")
    ax0.set_ylabel(r"${\rm Density}$")
    ax1.set_xlabel(r"$||\vec{\beta}||^2$")
    ax1.set_ylabel(r"${\rm Residuals}$")

    ax0.plot(
        x, chi2p, color="tab:blue", label=r"$\chi^2({\rm DoF}= " + f"{hist.dim}" + ")$"
    )
    ax0.legend(fontsize=16)
    ymin, ymax = ax0.get_ylim()
    ymin = chi2.pdf(hist.max_val, df=hist.dim)
    ax0.set_ylim([ymin, ymax])
    ax0.set_xlim([-0.5, hist_max_value + 0.5])

    ax0.text(
        0.0,
        ymax * 1.2,
        r"$p(\chi^2) = "
        + rf"{hist_pval_test.residuals_pvalue*100.:.1f}\%,\ "
        + r"p({\rm KS}) = "
        + rf"{hist_pval_test.kstest_pval*100:.1f}\%"
        + "$",
        color="darkred",
        fontsize=20,
    )

    for cl in confidence_level:
        p = chi2.isf(1.0 - cl, hist.dim)
        ax0.plot(
            [p] * 2,
            [ax0.get_ylim()[0], chi2.pdf(p, df=hist.dim)],
            color="tab:blue",
            linestyle="--",
            lw=1,
        )
        ax0.text(
            p,
            ymin * 1.2,
            rf"${cl*100:.0f}\% " + r"{\rm\ CL}$",
            ha="right",
            va="bottom",
            rotation=90,
            fontsize=20,
            color="darkred",
        )
        ax1.axvline(p, color="tab:blue", linestyle="--", zorder=0, lw=1)

    color = np.array(["gray"] * hist.nbins, dtype=object)
    pull = hist.pull
    color[(abs(pull) > 1.0) & (abs(pull) <= 2.0)] = "gold"
    color[(abs(pull) > 2.0) & (abs(pull) <= 3.0)] = "orangered"
    color[abs(pull) > 3.0] = "firebrick"
    ax1.bar(hist.bin_centers, hist.pull, width=hist.bin_width, color=color.tolist())
    ax1.set_ylim([-3.1, 3.1])

    return fig, (ax0, ax1)
