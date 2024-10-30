import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from .goodness_of_fit import Histogram


def chi2_analysis(
    deviations: np.ndarray,
    plot_name: str = None,
    event_prob_per_bin: float = None,
    **hist_kwargs,
) -> None:
    """
    _summary_

    Args:
        deviations (``np.ndarray``): deviations from normal distribution with mean 0 and sigma 1
        plot_name (``str``, default ``None``): name of the plot
        event_prob_per_bin (``float``, default ``None``): event probability at each bin. This arg
            will resize the bins to ensure each bin has the same event probability.
        hist_kwargs:
            bins (``int`` or ``np.ndarray``): if ``event_prob_per_bin`` is not given, number of bins
                are needed. Default 100.
            max_val (``int``, default ``20``): if ``bins`` are integer, max value is needed
            weights (``np.ndarray``, default ``None``): event weights. default is 1 per event.
    """
    if event_prob_per_bin is not None:
        bins = chi2.ppf(
            np.linspace(0, 1, int(np.ceil(1 / event_prob_per_bin)) + 1),
            df=deviations.shape[1],
        )
        bins = np.where(np.isinf(bins), hist_kwargs.get("max_val", 20.0), bins)
    else:
        bins = hist_kwargs.get("bins", 100)

    hist = Histogram(
        dim=deviations.shape[1],
        bins=bins,
        max_val=hist_kwargs.get("max_val", 20.0),
        vals=np.sum(deviations**2, axis=1),
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
    # ax0.set_xlim([-0.2, 20.2])
    # ax0.set_ylim([5e-4, 0.2])
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
    color[(abs(pull) > 1.0) & (abs(pull) <= 3.0)] = "gold"
    color[abs(pull) > 3.0] = "firebrick"
    ax1.bar(hist.bin_centers, hist.pull, width=hist.bin_width, color=color.tolist())
    ax1.set_ylim([-5, 5])

    if plot_name is not None:
        plt.savefig(plot_name)
    else:
        plt.show()
    plt.close()
