import numpy as np
import pytest

from nabu.transform_base import (
    _weighted_quantile,
    standardise_mean_std,
    standardise_median_quantile,
)

# 1-sigma tail quantile, matching the value used inside standardise_median_quantile.
from scipy.stats import norm

_Q = (1 - (norm.cdf(1) - norm.cdf(-1))) / 2


class TestStandardiseMeanStd:
    """Weighted mean/std standardisation."""

    def test_none_weights_match_unweighted(self):
        """Passing ``weights=None`` reproduces the unweighted statistics exactly."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(2000, 3))

        t_none, out_none = standardise_mean_std(data)
        t_ones, out_ones = standardise_mean_std(data, weights=np.ones(len(data)))

        # The unweighted path uses np.mean/np.std (ddof=0); the weighted path with
        # uniform weights must agree with it to floating-point precision.
        assert np.allclose(out_none, out_ones)
        assert np.allclose(
            t_none.to_dict()["shift-scale"]["shift"],
            t_ones.to_dict()["shift-scale"]["shift"],
        )
        assert np.allclose(
            t_none.to_dict()["shift-scale"]["scale"],
            t_ones.to_dict()["shift-scale"]["scale"],
        )

    def test_weighted_matches_unweighted_statistics(self):
        """Shift/scale equal the weighted mean and population weighted std."""
        rng = np.random.default_rng(1)
        data = rng.normal(loc=[1.0, -2.0], scale=[0.5, 3.0], size=(5000, 2))
        weights = rng.uniform(0.1, 2.0, size=len(data))

        transform, out = standardise_mean_std(data, weights=weights)

        exp_mean = np.average(data, axis=0, weights=weights)
        exp_std = np.sqrt(np.average((data - exp_mean) ** 2, axis=0, weights=weights))

        meta = transform.to_dict()["shift-scale"]
        assert np.allclose(meta["shift"], exp_mean)
        assert np.allclose(meta["scale"], exp_std)
        assert np.allclose(out, (data - exp_mean) / exp_std)

    def test_standardised_output_is_weight_centred_and_scaled(self):
        """The transformed data has weighted mean 0 and weighted std 1 per feature."""
        rng = np.random.default_rng(2)
        data = rng.normal(loc=[10.0, -5.0], scale=[2.0, 0.3], size=(8000, 2))
        weights = rng.uniform(0.0, 3.0, size=len(data))

        _, out = standardise_mean_std(data, weights=weights)

        w_mean = np.average(out, axis=0, weights=weights)
        w_std = np.sqrt(np.average((out - w_mean) ** 2, axis=0, weights=weights))
        assert np.allclose(w_mean, 0.0, atol=1e-10)
        assert np.allclose(w_std, 1.0)

    def test_integer_weights_match_expanded_duplicates(self):
        """Integer weights reproduce the statistics of the explicitly repeated sample."""
        rng = np.random.default_rng(3)
        data = rng.normal(size=(500, 2))
        counts = rng.integers(1, 6, size=len(data))

        expanded = np.repeat(data, counts, axis=0)

        _, out_expanded = standardise_mean_std(expanded)
        transform, _ = standardise_mean_std(data, weights=counts)

        meta = transform.to_dict()["shift-scale"]
        exp_mean = np.mean(expanded, axis=0)
        exp_std = np.std(expanded, axis=0)
        assert np.allclose(meta["shift"], exp_mean)
        assert np.allclose(meta["scale"], exp_std)
        # Sanity check the reference expansion really is centred/scaled the same way.
        assert np.allclose(out_expanded.mean(axis=0), 0.0, atol=1e-10)

    def test_weights_change_the_result(self):
        """Non-uniform weights shift the result away from the unweighted statistics."""
        rng = np.random.default_rng(4)
        data = rng.normal(loc=3.0, scale=1.0, size=(2000, 1))
        # Up-weight the upper tail so the weighted mean is clearly larger.
        weights = np.where(data[:, 0] > 3.0, 5.0, 1.0)

        _, out_unweighted = standardise_mean_std(data)
        transform, _ = standardise_mean_std(data, weights=weights)

        assert transform.to_dict()["shift-scale"]["shift"][0] > 3.0

    def test_shape_mismatch_raises(self):
        data = np.zeros((10, 2))
        with pytest.raises(ValueError, match="1D array"):
            standardise_mean_std(data, weights=np.ones((10, 2)))

    def test_length_mismatch_raises(self):
        data = np.zeros((10, 2))
        with pytest.raises(ValueError, match="same length"):
            standardise_mean_std(data, weights=np.ones(9))

    def test_negative_weights_raise(self):
        data = np.zeros((5, 2))
        weights = np.ones(5)
        weights[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            standardise_mean_std(data, weights=weights)

    def test_zero_total_weight_raises(self):
        data = np.zeros((5, 2))
        with pytest.raises(ValueError, match="total weight"):
            standardise_mean_std(data, weights=np.zeros(5))


class TestStandardiseMedianQuantile:
    """Weighted median / inter-quantile standardisation."""

    def test_none_weights_match_unweighted(self):
        """Passing ``weights=None`` reproduces the unweighted statistics exactly."""
        rng = np.random.default_rng(10)
        data = rng.normal(size=(2000, 3))

        t_none, out_none = standardise_median_quantile(data)
        t_ones, out_ones = standardise_median_quantile(data, weights=np.ones(len(data)))

        # The unweighted path uses np.median/np.quantile (linear); the weighted path
        # with uniform weights uses Hazen positions. The median coincides between the
        # two methods, but the inter-quantile scale differs slightly at finite N, so
        # only the shift is required to agree exactly here.
        assert np.allclose(
            t_none.to_dict()["shift-scale"]["shift"],
            t_ones.to_dict()["shift-scale"]["shift"],
        )

    def test_uniform_weights_match_hazen(self):
        """Uniform weights reproduce numpy's median and Hazen-method quantiles."""
        rng = np.random.default_rng(11)
        data = rng.normal(loc=[1.0, -2.0], scale=[0.5, 3.0], size=(5000, 2))

        transform, _ = standardise_median_quantile(data, weights=np.ones(len(data)))

        exp_median = np.median(data, axis=0)
        exp_scale = np.quantile(data, 1 - _Q, axis=0, method="hazen") - np.quantile(
            data, _Q, axis=0, method="hazen"
        )

        meta = transform.to_dict()["shift-scale"]
        assert np.allclose(meta["shift"], exp_median)
        assert np.allclose(meta["scale"], exp_scale)

    def test_weighted_median_analytic(self):
        """A hand-computed weighted median via Hazen interpolation."""
        data = np.array([[0.0], [1.0], [2.0], [3.0]])
        weights = np.array([1.0, 1.0, 1.0, 7.0])
        # Plotting positions p = (cumsum(w) - w/2) / sum(w) = [0.05, 0.15, 0.25, 0.65].
        # q = 0.5 interpolates between (0.25, value 2) and (0.65, value 3):
        #   2 + (0.5 - 0.25) / (0.65 - 0.25) * (3 - 2) = 2.625.
        transform, _ = standardise_median_quantile(data, weights=weights)
        assert np.isclose(transform.to_dict()["shift-scale"]["shift"][0], 2.625)

    def test_helper_clamps_outside_support(self):
        """Quantiles beyond the plotting positions clamp to the extreme samples."""
        data = np.array([10.0, 20.0, 30.0])
        weights = np.array([1.0, 1.0, 1.0])
        lo, hi = _weighted_quantile(data, [0.0, 1.0], weights)
        assert np.isclose(lo, 10.0)
        assert np.isclose(hi, 30.0)

    def test_zero_weight_samples_are_ignored(self):
        """Zero-weight samples do not affect the result."""
        rng = np.random.default_rng(12)
        data = rng.normal(size=(1000, 2))
        weights = rng.uniform(0.5, 2.0, size=len(data))

        # Append outliers carrying zero weight; they must be dropped.
        data_ext = np.vstack([data, np.full((50, 2), 1e6)])
        weights_ext = np.concatenate([weights, np.zeros(50)])

        t_ref, _ = standardise_median_quantile(data, weights=weights)
        t_ext, _ = standardise_median_quantile(data_ext, weights=weights_ext)

        assert np.allclose(
            t_ref.to_dict()["shift-scale"]["shift"],
            t_ext.to_dict()["shift-scale"]["shift"],
        )
        assert np.allclose(
            t_ref.to_dict()["shift-scale"]["scale"],
            t_ext.to_dict()["shift-scale"]["scale"],
        )

    def test_standardised_output_is_weight_centred(self):
        """The transformed data has weighted median 0 and unit weighted IQR per feature."""
        rng = np.random.default_rng(13)
        data = rng.normal(loc=[10.0, -5.0], scale=[2.0, 0.3], size=(8000, 2))
        weights = rng.uniform(0.0, 3.0, size=len(data))

        _, out = standardise_median_quantile(data, weights=weights)

        lo, med, hi = _weighted_quantile(out, [_Q, 0.5, 1 - _Q], weights)
        assert np.allclose(med, 0.0, atol=1e-10)
        assert np.allclose(hi - lo, 1.0)

    def test_weights_change_the_result(self):
        """Non-uniform weights shift the median away from the unweighted one."""
        rng = np.random.default_rng(14)
        data = rng.normal(loc=3.0, scale=1.0, size=(2000, 1))
        weights = np.where(data[:, 0] > 3.0, 5.0, 1.0)

        transform, _ = standardise_median_quantile(data, weights=weights)
        assert transform.to_dict()["shift-scale"]["shift"][0] > 3.0

    def test_length_mismatch_raises(self):
        data = np.zeros((10, 2))
        with pytest.raises(ValueError, match="same length"):
            standardise_median_quantile(data, weights=np.ones(9))

    def test_negative_weights_raise(self):
        data = np.zeros((5, 2))
        weights = np.ones(5)
        weights[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            standardise_median_quantile(data, weights=weights)

    def test_zero_total_weight_raises(self):
        data = np.zeros((5, 2))
        with pytest.raises(ValueError, match="total weight"):
            standardise_median_quantile(data, weights=np.zeros(5))
