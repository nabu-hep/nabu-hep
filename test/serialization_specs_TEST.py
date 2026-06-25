# Copyright (c) 2026 Danny van Dyk

"""Unit tests for the concrete header specs in :mod:`nabu.serialization.specs`.

Tests that only exercise parsing/serialisation and validation avoid importing
``flowjax``; tests that touch the registries (bijector/flow name validation) or
reconstruct runtime objects via ``build`` are gated behind
``pytest.importorskip("flowjax")``.
"""

from __future__ import annotations

import pytest

from nabu.serialization.base import SchemaError, SerializationError, UnknownTagError
from nabu.serialization.specs import (
    FORMAT_VERSION,
    BijectorSpec,
    CouplingFlowSpec,
    DalitzTransformSpec,
    FlowSpec,
    GenericFlowSpec,
    IdentityTransformSpec,
    LikelihoodSpec,
    MaskedAutoregressiveFlowSpec,
    PosteriorTransformSpec,
    ShiftScaleTransformSpec,
    UserDefinedTransformSpec,
)

# ---------------------------------------------------------------------------
# Reference headers (the worked example from doc/file-format.rst)
# ---------------------------------------------------------------------------

WORKED_EXAMPLE = {
    "model_type": "flow",
    "model": {
        "masked_autoregressive_flow": {
            "dim": 2,
            "transformer": None,
            "cond_dim": None,
            "flow_layers": 8,
            "nn_width": 50,
            "nn_depth": 1,
            "activation": "relu",
            "permutation": "reversed",
            "random_seed": 0,
        }
    },
    "posterior_transform": {
        "shift-scale": {
            "shift": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "log_axes": None,
            "log_shift": None,
            "log_scale": None,
        }
    },
    "version": "0.0.1",
}


def _minimal_header(**overrides):
    """A minimal but valid header (no transformer, identity transform)."""
    header = {
        "model_type": "flow",
        "model": {"masked_autoregressive_flow": {"dim": 2}},
        "posterior_transform": {},
    }
    header.update(overrides)
    return header


# ---------------------------------------------------------------------------
# Header round-trips (no flowjax needed: built-in tags dispatch directly)
# ---------------------------------------------------------------------------


class TestHeaderRoundTrip:
    def test_worked_example_round_trips(self):
        spec = LikelihoodSpec.from_dict(**WORKED_EXAMPLE)
        # Field types are recovered.
        assert isinstance(spec.model, MaskedAutoregressiveFlowSpec)
        assert spec.model.transformer is None
        assert isinstance(spec.posterior_transform, ShiftScaleTransformSpec)
        # Serialising back reproduces the header plus the new format_version.
        assert spec.to_dict() == {**WORKED_EXAMPLE, "format_version": FORMAT_VERSION}

    def test_missing_format_version_defaults(self):
        spec = LikelihoodSpec.from_dict(**_minimal_header())
        assert spec.format_version == FORMAT_VERSION

    def test_identity_transform_is_empty_object(self):
        spec = LikelihoodSpec.from_dict(**_minimal_header())
        assert isinstance(spec.posterior_transform, IdentityTransformSpec)
        assert spec.to_dict()["posterior_transform"] == {}

    def test_dalitz_transform_round_trips(self):
        header = _minimal_header(
            posterior_transform={
                "dalitz": {"md": 1.86483, "ma": 0.13957, "mb": 0.13957, "mc": 0.13957}
            }
        )
        spec = LikelihoodSpec.from_dict(**header)
        assert isinstance(spec.posterior_transform, DalitzTransformSpec)
        assert spec.to_dict()["posterior_transform"] == header["posterior_transform"]

    def test_shift_scale_with_log_axes_round_trips(self):
        header = _minimal_header(
            posterior_transform={
                "shift-scale": {
                    "shift": [1.0, 2.0],
                    "scale": [3.0, 4.0],
                    "log_axes": [1],
                    "log_shift": [0.5],
                    "log_scale": [2.0],
                }
            }
        )
        spec = LikelihoodSpec.from_dict(**header)
        assert spec.to_dict()["posterior_transform"] == header["posterior_transform"]

    def test_coupling_flow_nn_width_list_round_trips(self):
        # No transformer -> no bijector validation -> no flowjax import.
        header = _minimal_header(
            model={"coupling_flow": {"dim": 4, "nn_width": [50, 50]}}
        )
        spec = LikelihoodSpec.from_dict(**header)
        assert isinstance(spec.model, CouplingFlowSpec)
        assert spec.model.nn_width == [50, 50]
        assert spec.to_dict()["model"]["coupling_flow"]["nn_width"] == [50, 50]

    def test_generic_flow_spec_round_trips(self):
        spec = GenericFlowSpec(name="my_flow", kwargs={"dim": 3, "flow_layers": 2})
        assert spec.to_tagged() == {"my_flow": {"dim": 3, "flow_layers": 2}}


# ---------------------------------------------------------------------------
# Validation / error behaviour (no flowjax needed)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_non_flow_model_type(self):
        with pytest.raises(SchemaError, match="model_type"):
            LikelihoodSpec.from_dict(**_minimal_header(model_type="histogram"))

    def test_rejects_future_format_version(self):
        with pytest.raises(SchemaError, match="newer"):
            LikelihoodSpec.from_dict(**_minimal_header(format_version=FORMAT_VERSION + 1))

    def test_unexpected_flow_argument_reports_path(self):
        header = _minimal_header(
            model={"masked_autoregressive_flow": {"dim": 2, "bogus": 1}}
        )
        with pytest.raises(SchemaError, match="unexpected key"):
            LikelihoodSpec.from_dict(**header)

    def test_user_defined_transform_refuses_to_serialise(self):
        with pytest.raises(SerializationError, match="user-defined"):
            UserDefinedTransformSpec().to_tagged()

    def test_user_defined_transform_refuses_to_build(self):
        with pytest.raises(SerializationError, match="user-defined"):
            UserDefinedTransformSpec().build()

    def test_posterior_transform_non_mapping_rejected(self):
        with pytest.raises(SchemaError):
            PosteriorTransformSpec.from_tagged(["shift-scale"])

    def test_bijector_spec_rejects_multi_key_transformer(self):
        # Shape validation happens before the (lazy) flowjax import, so no
        # registry access is needed for these malformed inputs.
        with pytest.raises(SchemaError, match="exactly one"):
            BijectorSpec.from_tagged({"Affine": {}, "Tanh": {}})

    def test_bijector_spec_rejects_non_object_payload(self):
        with pytest.raises(SchemaError, match="must be an object"):
            BijectorSpec.from_tagged({"Affine": 5})


# ---------------------------------------------------------------------------
# Transform reconstruction (needs nabu.transform_base; jax already imported)
# ---------------------------------------------------------------------------


class TestTransformBuild:
    def test_identity_builds_identity(self):
        from nabu.transform_base import PosteriorTransform

        transform = IdentityTransformSpec().build()
        assert isinstance(transform, PosteriorTransform)
        assert transform.to_dict() == {}

    def test_shift_scale_round_trips_through_runtime(self):
        spec = ShiftScaleTransformSpec(shift=[0.0, 0.0], scale=[1.0, 1.0])
        assert spec.build().to_dict() == spec.to_tagged()

    def test_dalitz_round_trips_through_runtime(self):
        spec = DalitzTransformSpec(md=1.86483, ma=0.13957, mb=0.13957, mc=0.13957)
        assert spec.build().to_dict() == spec.to_tagged()


# ---------------------------------------------------------------------------
# Registry-backed validation and build (require flowjax)
# ---------------------------------------------------------------------------


class TestFlowjaxBacked:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("flowjax")

    def test_transformer_bijector_round_trips(self):
        header = _minimal_header(
            model={
                "coupling_flow": {
                    "dim": 4,
                    "transformer": {
                        "RationalQuadraticSpline": {"knots": 8, "interval": 4}
                    },
                }
            }
        )
        spec = LikelihoodSpec.from_dict(**header)
        assert isinstance(spec.model.transformer, BijectorSpec)
        assert spec.model.transformer.name == "RationalQuadraticSpline"
        assert (
            spec.to_dict()["model"]["coupling_flow"]["transformer"]
            == header["model"]["coupling_flow"]["transformer"]
        )

    def test_unknown_bijector_raises(self):
        with pytest.raises(UnknownTagError, match="unknown bijector"):
            BijectorSpec.from_tagged({"NotABijector": {}})

    def test_unknown_flow_raises(self):
        with pytest.raises(UnknownTagError, match="unknown flow"):
            FlowSpec.from_tagged({"not_a_flow": {}})

    def test_generic_flow_build_rejects_malformed_transformer(self):
        # GenericFlowSpec.build validates the transformer shape through
        # BijectorSpec, raising a structured SchemaError rather than a raw
        # ValueError/TypeError from unpacking.
        spec = GenericFlowSpec(
            name="masked_autoregressive_flow",
            kwargs={"dim": 2, "transformer": {"Affine": {}, "Tanh": {}}},
        )
        with pytest.raises(SchemaError, match="exactly one"):
            spec.build(random_seed=0)

    def test_build_skeleton_produces_flow_likelihood(self):
        spec = LikelihoodSpec.from_dict(**_minimal_header())
        likelihood = spec.build_skeleton(random_seed=0)
        assert likelihood.model_type == "flow"
        assert hasattr(likelihood.model, "log_prob")

    def test_masked_flow_builds(self):
        spec = MaskedAutoregressiveFlowSpec(dim=2, flow_layers=2)
        likelihood = spec.build(random_seed=0)
        assert likelihood.model_type == "flow"
