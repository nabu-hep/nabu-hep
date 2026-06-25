# Copyright (c) 2026 Danny van Dyk

"""Integration tests for ``Likelihood.save``/``load`` and ``to_spec``.

Header-validation tests stay free of ``flowjax``; tests that build a real flow,
write it to disk and read it back are gated behind
``pytest.importorskip("flowjax")``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import nabu
from nabu.serialization import FORMAT_VERSION, SchemaError, SerializationError


# ---------------------------------------------------------------------------
# Load validation (no flowjax: errors are raised before any flow is built)
# ---------------------------------------------------------------------------


class TestLoadValidation:
    def test_rejects_non_nabu_suffix(self, tmp_path):
        path = tmp_path / "model.txt"
        path.write_text("{}")
        with pytest.raises(SerializationError, match="existing .nabu file"):
            nabu.Likelihood.load(str(path))

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(SerializationError, match="existing .nabu file"):
            nabu.Likelihood.load(str(tmp_path / "absent.nabu"))

    def test_rejects_non_flow_header(self, tmp_path):
        path = tmp_path / "model.nabu"
        header = {
            "model_type": "histogram",
            "model": {"masked_autoregressive_flow": {"dim": 2}},
            "posterior_transform": {},
        }
        with open(path, "wb") as handle:
            handle.write((json.dumps(header) + "\n").encode())
        with pytest.raises(SchemaError, match="model_type"):
            nabu.Likelihood.load(str(path))

    def test_rejects_unknown_flow_argument(self, tmp_path):
        path = tmp_path / "model.nabu"
        header = {
            "model_type": "flow",
            "model": {"masked_autoregressive_flow": {"dim": 2, "bogus": 1}},
            "posterior_transform": {},
        }
        with open(path, "wb") as handle:
            handle.write((json.dumps(header) + "\n").encode())
        with pytest.raises(SchemaError, match="unexpected key"):
            nabu.Likelihood.load(str(path))


# ---------------------------------------------------------------------------
# Full save/load round-trips (require flowjax)
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("flowjax")

    def _model(self):
        from nabu.flow import masked_autoregressive_flow

        return masked_autoregressive_flow(dim=2, flow_layers=2)

    def test_header_contains_format_version(self, tmp_path):
        path = tmp_path / "model.nabu"
        self._model().save(str(path))
        with open(path, "rb") as handle:
            header = json.loads(handle.readline().decode())
        assert header["model_type"] == "flow"
        assert header["format_version"] == FORMAT_VERSION
        assert header["posterior_transform"] == {}

    def test_round_trip_preserves_leaves(self, tmp_path):
        likelihood = self._model()
        path = tmp_path / "model.nabu"
        likelihood.save(str(path))
        loaded = nabu.Likelihood.load(str(path), random_seed=0)

        assert loaded.model_type == "flow"
        x = np.zeros((4, 2))
        assert np.allclose(loaded.log_prob(x), likelihood.log_prob(x))
        # The reconstructed architecture description matches the original.
        assert loaded.to_spec().to_dict() == likelihood.to_spec().to_dict()

    def test_round_trip_preserves_shift_scale_transform(self, tmp_path):
        from nabu.transform_base import standardise_mean_std

        data = np.random.RandomState(0).normal(size=(100, 2))
        transform, _ = standardise_mean_std(data)
        likelihood = self._model()
        likelihood.transform = transform

        path = tmp_path / "model.nabu"
        likelihood.save(str(path))
        loaded = nabu.Likelihood.load(str(path), random_seed=0)

        assert loaded.transform.to_dict() == transform.to_dict()

    def test_user_defined_transform_refuses_to_save(self, tmp_path):
        from nabu.transform_base import PosteriorTransform

        likelihood = self._model()
        likelihood.transform = PosteriorTransform(
            "user-defined", forward=lambda x: x, backward=lambda x: x
        )
        with pytest.raises(SerializationError, match="user-defined"):
            likelihood.save(str(tmp_path / "model.nabu"))
