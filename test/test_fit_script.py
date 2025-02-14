"""Test fit script output"""
import numpy as np
from scipy.stats import chi2

import nabu
from nabu.goodness_of_fit import Histogram


def test_fit_script_output():
    """Test fit script output"""

    try:
        lm = nabu.Likelihood.load("./test/results/TEST-RESULT-COMPLETE/model.nabu")

        lm_chi2 = float(lm.chi2(np.array([0.0, 0.0])))
        assert np.isclose(lm_chi2, 14.865792274475098), f"chi^2 value is wrong, {chi2}"
    except FileNotFoundError:
        assert False, "Model not found. Please execute pytest from the main folder."

    try:
        deviations = np.load("./test/results/TEST-RESULT-COMPLETE/deviations.npz")[
            "deviations"
        ]
    except FileNotFoundError:
        assert False, "Deviations not found. Please execute pytest from the main folder."

    bins = chi2.ppf(
        np.linspace(0.0, 1.0, int(np.ceil(1.0 / 0.1)) + 1), df=deviations.shape[1]
    )
    chi2_test = np.sum(deviations**2, axis=1)

    hist = Histogram(dim=deviations.shape[1], bins=bins, vals=chi2_test)

    assert np.isclose(
        float(hist.residuals_pvalue), 0.016930612958101166
    ), f"p-val for residuals are wrong, {hist.residuals_pvalue}"
    assert np.isclose(
        float(hist.kstest_pval), 0.023394529807423625
    ), f"p-val for KST is wrong, {hist.kstest_pval}"
