import numpy as np
import pytest

from nabu.transform_base import standardise_mean_std


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
