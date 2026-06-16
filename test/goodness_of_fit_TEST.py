"""Tests for the weighted goodness-of-fit / Kolmogorov-Smirnov test (proposal A)."""
import numpy as np
import pytest
from scipy.stats import chi2, kstest

from nabu.goodness_of_fit import Histogram, weighted_kstest


def _chi2_cdf(dim):
    return lambda x: chi2.cdf(x, df=dim)


class TestKSTestPVal:
    """Weighted one-sample KS test against the chi^2 distribution."""

    def test_uniform_weights_match_unweighted_ks(self):
        """With uniform weights the test reduces to scipy's exact one-sample KS."""
        rng = np.random.default_rng(0)
        dim = 4
        vals = chi2.rvs(df=dim, size=5000, random_state=rng)

        d, pval, n_eff = weighted_kstest(vals, _chi2_cdf(dim))

        # scipy's default (exact at this N) one-sample KS is the reference: equal
        # weights must reproduce it exactly.
        ref = kstest(vals, _chi2_cdf(dim))
        assert np.isclose(d, ref.statistic), "KS statistic D disagrees with scipy"
        assert np.isclose(pval, ref.pvalue, rtol=1e-9), "p-value disagrees with scipy"
        assert np.isclose(
            n_eff, len(vals)
        ), "effective N must equal N for uniform weights"

    def test_none_weights_equivalent_to_ones(self):
        """Passing ``None`` weights is identical to passing an array of ones."""
        rng = np.random.default_rng(1)
        vals = chi2.rvs(df=3, size=2000, random_state=rng)

        assert weighted_kstest(vals, _chi2_cdf(3), None) == weighted_kstest(
            vals, _chi2_cdf(3), np.ones(len(vals))
        )

    def test_weighted_statistic_matches_expanded_duplicates(self):
        """Integer weights on unique points reproduce the D of the expanded sample,
        but the p-value uses the (smaller) effective N rather than the row count."""
        rng = np.random.default_rng(2)
        unique_vals = chi2.rvs(df=5, size=400, random_state=rng)
        counts = rng.integers(1, 6, size=unique_vals.shape)

        expanded = np.repeat(unique_vals, counts)

        d_w, p_w, n_eff = weighted_kstest(unique_vals, _chi2_cdf(5), counts)
        ref = kstest(expanded, _chi2_cdf(5), method="asymp")

        # Same empirical CDF -> identical KS statistic.
        assert np.isclose(
            d_w, ref.statistic
        ), "weighted D must match the expanded-sample D"

        # The effective sample size is below the number of rows whenever weights vary,
        # so the (correct) weighted p-value is *less* significant than the over-powered
        # unweighted test that treats every duplicated row as an independent draw.
        assert n_eff < len(expanded)
        assert (
            p_w > ref.pvalue
        ), "weighted test must be less over-powered than expanded KS"

    def test_histogram_kstest_pval_uses_weights(self):
        """Histogram.kstest_pval / kstest_dval / effective_nobs honour the weights."""
        rng = np.random.default_rng(3)
        dim = 4
        vals = chi2.rvs(df=dim, size=1500, random_state=rng)
        weights = rng.uniform(0.1, 2.0, size=vals.shape)

        bins = chi2.ppf(np.linspace(0.0, 1.0, 11), df=dim)
        hist = Histogram(dim=dim, bins=bins, vals=vals, weights=weights)

        d, pval, n_eff = weighted_kstest(vals, _chi2_cdf(dim), weights)
        assert np.isclose(hist.kstest_pval, pval)
        assert np.isclose(hist.kstest_dval, d)
        assert np.isclose(hist.effective_nobs, n_eff)
        assert n_eff < len(vals), "non-uniform weights must reduce the effective N"

    def test_histogram_accepts_array_weights_without_error(self):
        """Regression: array weights previously raised an ambiguous-truth ValueError."""
        vals = np.linspace(0.5, 20.0, 50)
        weights = np.linspace(0.5, 1.5, 50)
        hist = Histogram(dim=3, bins=10, vals=vals, max_val=25.0, weights=weights)
        assert np.allclose(hist.weights, weights)

    def test_invalid_inputs_raise_value_error(self):
        """Degenerate inputs raise ValueError rather than producing NaNs."""
        vals = np.array([1.0, 2.0, 3.0])

        # length mismatch
        with pytest.raises(ValueError):
            weighted_kstest(vals, _chi2_cdf(2), np.ones(2))
        # empty sample
        with pytest.raises(ValueError):
            weighted_kstest(np.array([]), _chi2_cdf(2))
        # negative weight
        with pytest.raises(ValueError):
            weighted_kstest(vals, _chi2_cdf(2), np.array([1.0, -1.0, 1.0]))
        # non-positive total weight
        with pytest.raises(ValueError):
            weighted_kstest(vals, _chi2_cdf(2), np.zeros(3))

    def test_histogram_invalid_weights_raise_value_error(self):
        """Histogram validates weights instead of asserting / dividing by zero."""
        vals = np.linspace(0.5, 20.0, 10)
        bins = chi2.ppf(np.linspace(0.0, 1.0, 6), df=3)

        # length mismatch (previously a bare assert)
        with pytest.raises(ValueError):
            Histogram(dim=3, bins=bins, vals=vals, weights=np.ones(5))
        # negative weight
        with pytest.raises(ValueError):
            Histogram(dim=3, bins=bins, vals=vals, weights=-np.ones(10))
        # non-positive total weight -> would make bin_weights inf/nan
        with pytest.raises(ValueError):
            Histogram(dim=3, bins=bins, vals=vals, weights=np.zeros(10))
