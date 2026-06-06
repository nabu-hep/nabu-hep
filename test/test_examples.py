"""Tests for the ATLAS, LHCb, and WET example models."""

import os

import jax.numpy as jnp
import numpy as np
import pytest

import nabu

_TEST_DIR = os.path.dirname(__file__)

_ATLAS_MODEL = os.path.join(_TEST_DIR, "ATLAS", "atlas-model.nabu")
_ATLAS_DATA = os.path.join(_TEST_DIR, "ATLAS", "atlas-dataset.npz")
_LHCB_MODEL = os.path.join(_TEST_DIR, "LHCb", "lhcb-model.nabu")
_LHCB_DATA = os.path.join(_TEST_DIR, "LHCb", "lhcb-dataset.npz")
_WET_MODEL = os.path.join(_TEST_DIR, "WET", "wet-model.nabu")
_WET_DATA = os.path.join(_TEST_DIR, "WET", "wet-dataset.npz")


def _require_file(path):
    if not os.path.isfile(path):
        pytest.skip(f"Example file not found: {path}")


# ---------------------------------------------------------------------------
# ATLAS
# ---------------------------------------------------------------------------


class TestATLAS:
    @classmethod
    def setup_class(cls):
        _require_file(_ATLAS_MODEL)
        _require_file(_ATLAS_DATA)
        cls.lm = nabu.Likelihood.load(_ATLAS_MODEL)
        with np.load(_ATLAS_DATA) as data:
            cls.X_test = data["X_test"]

    def test_load(self):
        assert self.lm is not None

    def test_dimension(self):
        assert self.X_test.shape[1] == 5

    def test_log_prob_shape(self):
        lp = np.array(self.lm.model.log_prob(jnp.array(self.X_test[:10])))
        assert lp.shape == (10,)

    def test_log_prob_finite(self):
        lp = np.array(self.lm.model.log_prob(jnp.array(self.X_test[:100])))
        assert np.all(np.isfinite(lp))

    def test_log_prob_values(self):
        lp = np.array(self.lm.model.log_prob(jnp.array(self.X_test[:5])))
        expected = np.array(
            [
                4.983366012573242,
                9.382678985595703,
                9.323177337646484,
                4.612287521362305,
                4.696924209594727,
            ]
        )
        assert np.allclose(
            lp, expected, rtol=1e-4
        ), f"log_prob values differ from expected.\nGot:      {lp}\nExpected: {expected}"

    def test_sample_shape(self):
        samples = self.lm.sample(20)
        assert samples.shape == (20, 5)


# ---------------------------------------------------------------------------
# LHCb
# ---------------------------------------------------------------------------


class TestLHCb:
    @classmethod
    def setup_class(cls):
        _require_file(_LHCB_MODEL)
        _require_file(_LHCB_DATA)
        cls.lm = nabu.Likelihood.load(_LHCB_MODEL)
        with np.load(_LHCB_DATA) as data:
            cls.X_test = data["X_test"]

    def test_load(self):
        assert self.lm is not None

    def test_dimension(self):
        assert self.X_test.shape[1] == 2

    def test_log_prob_shape(self):
        lp = np.array(self.lm.model.log_prob(jnp.array(self.X_test[:10])))
        assert lp.shape == (10,)

    def test_log_prob_finite(self):
        lp = np.array(self.lm.model.log_prob(jnp.array(self.X_test)))
        assert np.all(np.isfinite(lp))

    def test_log_prob_values(self):
        lp = np.array(self.lm.model.log_prob(jnp.array(self.X_test[:5])))
        expected = np.array(
            [
                1.1209361553192139,
                -0.24916648864746094,
                -0.978630781173706,
                1.051612138748169,
                0.2813303470611572,
            ]
        )
        assert np.allclose(
            lp, expected, rtol=1e-4
        ), f"log_prob values differ from expected.\nGot:      {lp}\nExpected: {expected}"

    def test_sample_shape(self):
        samples = self.lm.sample(20)
        assert samples.shape == (20, 2)


# ---------------------------------------------------------------------------
# WET
# ---------------------------------------------------------------------------


class TestWET:
    @classmethod
    def setup_class(cls):
        _require_file(_WET_MODEL)
        _require_file(_WET_DATA)
        cls.lm = nabu.Likelihood.load(_WET_MODEL)
        with np.load(_WET_DATA) as data:
            cls.X_test = data["X_test"]

    def test_load(self):
        assert self.lm is not None

    def test_dimension(self):
        assert self.X_test.shape[1] == 5

    def test_log_prob_shape(self):
        lp = np.array(self.lm.model.log_prob(jnp.array(self.X_test[:10])))
        assert lp.shape == (10,)

    def test_log_prob_finite(self):
        lp = np.array(self.lm.model.log_prob(jnp.array(self.X_test[:100])))
        assert np.all(np.isfinite(lp))

    def test_log_prob_values(self):
        lp = np.array(self.lm.model.log_prob(jnp.array(self.X_test[:5])))
        expected = np.array(
            [
                -4.921694755554199,
                -2.675860643386841,
                -2.5456180572509766,
                -3.853290557861328,
                -1.888563632965088,
            ]
        )
        assert np.allclose(
            lp, expected, rtol=1e-4
        ), f"log_prob values differ from expected.\nGot:      {lp}\nExpected: {expected}"

    def test_sample_shape(self):
        samples = self.lm.sample(20)
        assert samples.shape == (20, 5)
