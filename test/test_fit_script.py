"""Test fit script output"""
import numpy as np
from scipy.stats import chi2

from nabu import Histogram


def test_fit_script_output():
    """Test fit script output"""
    try:
        deviations = np.load("./test/results/TEST-RESULT/deviations.npz")["deviations"]
    except FileNotFoundError:
        assert False, "Please execute pytest from the main folder"

    bins = chi2.ppf(
        np.linspace(0.0, 1.0, int(np.ceil(1.0 / 0.1)) + 1), df=deviations.shape[1]
    )
    chi2_test = np.sum(deviations**2, axis=1)

    hist = Histogram(dim=deviations.shape[1], bins=bins, vals=chi2_test)

    assert np.isclose(
        hist.residuals_pvalue, 0.019739547852598904
    ), f"p-val for residuals are wrong, {hist.residuals_pvalue}"
    assert np.isclose(
        hist.kstest_pval, 0.006416630663697942
    ), f"p-val for KST is wrong, {hist.kstest_pval}"
