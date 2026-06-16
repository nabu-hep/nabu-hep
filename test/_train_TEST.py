"""Tests for weighted maximum-likelihood training (proposal C)."""
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from flowjax import wrappers

from nabu.flow._flows import masked_autoregressive_flow
from nabu.flow._train import MaximumLikelihoodLoss
from nabu.flow.metrics import GoodnessOfFit


def _partition(dist):
    return eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, wrappers.NonTrainable),
    )


class TestWeightedTraining:
    """Weighted negative-log-likelihood loss and the fit_to_data plumbing."""

    def test_uniform_weights_match_unweighted_loss(self):
        """weighted=True with unit weights reproduces the plain mean NLL."""
        lh = masked_autoregressive_flow(dim=2, flow_layers=2, nn_width=8, random_seed=0)
        params, static = _partition(lh.model)
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.normal(size=(256, 2)))

        weighted = MaximumLikelihoodLoss(weighted=True)
        unweighted = MaximumLikelihoodLoss(weighted=False)
        assert np.isclose(
            float(weighted(params, static, x, jnp.ones(len(x)))),
            float(unweighted(params, static, x)),
        )

    def test_integer_weights_match_expanded_sample(self):
        """Integer weights on unique points equal the mean NLL of the repeated sample.

        This is the defining property of the weighted MLE: it removes the need to
        unweight by physically resampling (which duplicates rows)."""
        lh = masked_autoregressive_flow(dim=2, flow_layers=2, nn_width=8, random_seed=0)
        params, static = _partition(lh.model)
        rng = np.random.default_rng(1)
        unique_x = rng.normal(size=(80, 2))
        counts = rng.integers(1, 6, size=len(unique_x))
        expanded = np.repeat(unique_x, counts, axis=0)

        weighted = MaximumLikelihoodLoss(weighted=True)
        unweighted = MaximumLikelihoodLoss(weighted=False)
        loss_w = float(
            weighted(params, static, jnp.asarray(unique_x), jnp.asarray(counts, float))
        )
        loss_u = float(unweighted(params, static, jnp.asarray(expanded)))
        assert np.isclose(loss_w, loss_u, rtol=1e-5)

    def test_fit_to_data_weighted_runs_with_metric(self):
        """fit_to_data(weights=...) trains and forwards weights to the GoF metric."""
        lh = masked_autoregressive_flow(dim=1, flow_layers=2, nn_width=8, random_seed=0)
        rng = np.random.default_rng(2)
        data = rng.normal(size=(400, 1))
        weights = rng.uniform(0.2, 2.0, size=400)

        history = lh.fit_to_data(
            data,
            weights=weights,
            max_epochs=3,
            batch_size=100,
            validation_probability=0.2,
            verbose=False,
            metrics=[GoodnessOfFit(prob_per_bin=0.25)],
            random_seed=0,
        )
        assert np.all(np.isfinite(history["train"]))
        assert np.all(np.isfinite(history["val"]))
        # the weighted GoF metric was recorded each epoch
        assert "train_kstest_pvalue" in history
        assert "val_kstest_pvalue" in history

    def test_weights_is_keyword_only(self):
        """weights must be a keyword-only argument of fit_to_data (and fit)."""
        import inspect

        from nabu.flow._flow_likelihood import FlowLikelihood
        from nabu.flow._train import fit

        for func in (FlowLikelihood.fit_to_data, fit):
            kind = inspect.signature(func).parameters["weights"].kind
            assert kind is inspect.Parameter.KEYWORD_ONLY, func.__qualname__

    def test_weighted_fit_follows_the_weights(self):
        """Down-weighting one mode pulls the fitted density toward the other."""
        rng = np.random.default_rng(3)
        pos = rng.normal(3.0, 0.3, size=(600, 1))
        neg = rng.normal(-3.0, 0.3, size=(600, 1))
        data = np.vstack([pos, neg])
        # essentially switch off the negative cluster
        weights = np.concatenate([np.ones(600), np.full(600, 1e-3)])

        lh = masked_autoregressive_flow(dim=1, flow_layers=4, nn_width=16, random_seed=0)
        lh.fit_to_data(
            data,
            weights=weights,
            max_epochs=200,
            max_patience=20,
            batch_size=128,
            validation_probability=0.2,
            verbose=False,
            random_seed=0,
        )
        lp_pos = float(lh.log_prob(np.array([[3.0]])))
        lp_neg = float(lh.log_prob(np.array([[-3.0]])))
        assert lp_pos > lp_neg, "weighted fit should favour the up-weighted mode"
