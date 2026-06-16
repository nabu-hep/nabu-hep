"""Tests for weighted goodness-of-fit threading in the Likelihood API (proposal B)."""
import numpy as np
from scipy.stats import chi2

from nabu.likelihood import Likelihood
from nabu.goodness_of_fit import weighted_kstest


class _StubLikelihood(Likelihood):
    """Minimal concrete Likelihood whose ``chi2`` returns precomputed values.

    This isolates the goodness-of-fit threading from the flow model so the test
    does not depend on deserialising a trained flow.
    """

    model_type = "stub"

    def __init__(self, chi2_vals):
        self._chi2_vals = np.asarray(chi2_vals, dtype=float)

    def chi2(self, x):  # noqa: ARG002 - input ignored, values are precomputed
        return self._chi2_vals

    def inverse(self):
        raise NotImplementedError

    def fit_to_data(self, *args, **kwargs):
        raise NotImplementedError

    def to_dict(self):
        return {}


class TestWeightedGoodnessOfFit:
    """Likelihood.kstest_pvalue / goodness_of_fit / is_valid honour weights."""

    def _setup(self, seed, dim=4, size=2000):
        rng = np.random.default_rng(seed)
        chi2_vals = chi2.rvs(df=dim, size=size, random_state=rng)
        weights = rng.uniform(0.1, 2.0, size=chi2_vals.shape)
        lh = _StubLikelihood(chi2_vals)
        test_data = np.zeros((size, dim))  # only the trailing shape (dim) is used
        return lh, test_data, weights, chi2_vals, dim

    def test_kstest_pvalue_matches_weighted_kstest(self):
        lh, test_data, weights, chi2_vals, dim = self._setup(0)
        _, expected, _ = weighted_kstest(
            chi2_vals, lambda x: chi2.cdf(x, df=dim), weights
        )
        assert np.isclose(lh.kstest_pvalue(test_data, weights=weights), expected)

    def test_kstest_pvalue_defaults_to_unweighted(self):
        lh, test_data, _, chi2_vals, dim = self._setup(1)
        _, expected, _ = weighted_kstest(chi2_vals, lambda x: chi2.cdf(x, df=dim))
        assert np.isclose(lh.kstest_pvalue(test_data), expected)

    def test_goodness_of_fit_forwards_weights(self):
        lh, test_data, weights, chi2_vals, dim = self._setup(2)
        hist = lh.goodness_of_fit(test_data, prob_per_bin=0.1, weights=weights)
        assert np.allclose(hist.weights, weights)

        _, expected, n_eff = weighted_kstest(
            chi2_vals, lambda x: chi2.cdf(x, df=dim), weights
        )
        assert np.isclose(hist.kstest_pval, expected)
        assert np.isclose(hist.effective_nobs, n_eff)
        assert n_eff < len(chi2_vals), "non-uniform weights must reduce the effective N"

    def test_is_valid_accepts_weights(self):
        # chi2 values genuinely drawn from chi2(dim) -> should pass a loose threshold.
        lh, test_data, weights, _, _ = self._setup(3)
        assert lh.is_valid(test_data, threshold=0.01, weights=weights) is True

        # Shifted values are a poor fit -> should fail regardless of weights.
        bad = _StubLikelihood(lh._chi2_vals + 8.0)
        assert bad.is_valid(test_data, threshold=0.05, weights=weights) is False
