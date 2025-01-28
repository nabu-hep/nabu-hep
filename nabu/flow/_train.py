import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from flowjax import wrappers
from flowjax.distributions import AbstractDistribution
from flowjax.train.train_utils import count_fruitless, get_batches, step, train_val_split
from flowjax.wrappers import unwrap
from jax.tree_util import tree_leaves
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, PyTree
from tqdm import tqdm

from nabu.tensorboard import SummaryWriter

__all__ = ["get_optimizer", "fit"]


def __dir__():
    return __all__


def get_optimizer(opt: str):
    """Retreive the optimiser"""
    return {"adam": optax.adam, "sgd": optax.sgd, "adagrad": optax.adagrad}[opt]


def _append_metric(
    metric: callable, history: dict[str, list[float]], tag: str
) -> dict[str, list[float]]:
    """add metric to history"""
    for key, item in metric.items():
        name = tag + "_" + key
        if name in history:
            history[name].append(item)
        else:
            history.update({name: [item]})
    return history


class MaximumLikelihoodLoss:
    """Loss for fitting a flow with maximum likelihood (negative log likelihood).

    This loss can be used to learn either conditional or unconditional distributions.
    """

    def __init__(self, l1: float = 0.0, l2: float = 0.0):
        self.l1 = l1
        self.l2 = l2

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        x: Array,
        condition: Array = None,
        key: PRNGKeyArray = None,
    ) -> Float[Array, ""]:
        """Compute the loss. Key is ignored (for consistency of API)."""
        dist = unwrap(eqx.combine(params, static))
        nll = -dist.log_prob(x, condition).mean()

        if self.l2 != 0.0:
            nll += self.l2 * sum(jnp.sum(jnp.square(p)) for p in tree_leaves(params))
        if self.l1 != 0.0:
            nll += self.l1 * sum(jnp.sum(jnp.abs(p)) for p in tree_leaves(params))

        return nll


def fit(
    key: PRNGKeyArray,
    dist: PyTree,
    x: ArrayLike,
    L1_regularisation_coef: float = 0.0,
    L2_regularisation_coef: float = 0.0,
    condition: ArrayLike = None,
    optimizer: optax.GradientTransformation = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    check_every: int = 1,
    batch_size: int = 100,
    val_prop: float = 0.1,
    return_best: bool = True,
    show_progress: bool = True,
    lr_scheduler=None,
    plot_progress: str = None,
    metrics: list[callable] = None,
    log: str = None,
):
    r"""Train a PyTree (e.g. a distribution) to samples from the target.

    The model can be unconditional :math:`p(x)` or conditional
    :math:`p(x|\text{condition})`. Note that the last batch in each epoch is dropped
    if truncated (to avoid recompilation). This function can also be used to fit
    non-distribution pytrees as long as a compatible loss function is provided.

    Args:
        key: Jax random seed.
        dist: The pytree to train (usually a distribution).
        x: Samples from target distribution.
        learning_rate: The learning rate for adam optimizer. Ignored if optimizer is
            provided.
        optimizer: Optax optimizer. Defaults to None.
        condition: Conditioning variables. Defaults to None.
        loss_fn: Loss function. Defaults to MaximumLikelihoodLoss.
        max_epochs: Maximum number of epochs. Defaults to 100.
        max_patience: Number of consecutive epochs with no validation loss improvement
            after which training is terminated. Defaults to 5.
        batch_size: Batch size. Defaults to 100.
        val_prop: Proportion of data to use in validation set. Defaults to 0.1.
        return_best: Whether the result should use the parameters where the minimum loss
            was reached (when True), or the parameters after the last update (when
            False). Defaults to True.
        show_progress: Whether to show progress bar. Defaults to True.
        lr_scheduler: Learning rate scheduler. Defaults to None.
        plot_progress: Name of the monitoring plots. If given: plot the model once in a while to monitor progress visually. Defaults to None.

    Returns:
        A tuple containing the trained distribution and the losses.
    """
    data = (x,) if condition is None else (x, condition)
    data = tuple(jnp.asarray(a) for a in data)

    loss_fn = MaximumLikelihoodLoss(l1=L1_regularisation_coef, l2=L2_regularisation_coef)

    params, static = eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, wrappers.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    # train val split
    key, subkey = jr.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_prop)
    losses = {
        "train": [],
        "val": [],
        "lr": [float(opt_state.hyperparams["learning_rate"])],
    }

    metrics = metrics or []

    loop = tqdm(
        range(max_epochs),
        disable=not show_progress,
        unit="epoch",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    )

    train_summary = SummaryWriter(log, "train")
    val_summary = SummaryWriter(log, "val")

    try:
        for epoch in loop:
            # Shuffle data
            key, *subkeys = jr.split(key, 3)
            train_data = [jr.permutation(subkeys[0], a) for a in train_data]
            val_data = [jr.permutation(subkeys[1], a) for a in val_data]

            # Train epoch
            batch_losses = []
            for batch in zip(*get_batches(train_data, batch_size)):
                key, subkey = jr.split(key)
                params, opt_state, loss_i = step(
                    params,
                    static,
                    *batch,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    loss_fn=loss_fn,
                    key=subkey,
                )
                batch_losses.append(loss_i)
            losses["train"].append((sum(batch_losses) / len(batch_losses)).item())
            train_summary.scalar("loss", losses["train"][-1], epoch)
            for metric in metrics:
                losses = _append_metric(
                    metric(eqx.combine(params, static), train_data[0]), losses, "train"
                )

            # Val epoch
            batch_losses = []
            for batch in zip(*get_batches(val_data, batch_size)):
                key, subkey = jr.split(key)
                loss_i = loss_fn(params, static, *batch, key=subkey)
                batch_losses.append(loss_i)
            losses["val"].append((sum(batch_losses) / len(batch_losses)).item())
            val_summary.scalar("loss", losses["val"][-1], epoch)
            for metric in metrics:
                losses = _append_metric(
                    metric(eqx.combine(params, static), val_data[0]), losses, "val"
                )

            if lr_scheduler is not None:
                opt_state.hyperparams["learning_rate"] = lr_scheduler(epoch + 1)
                losses["lr"].append(float(opt_state.hyperparams["learning_rate"]))
                train_summary.scalar("learning_rate", losses["lr"][-1], epoch)

            loop.set_postfix({k: v[-1] for k, v in losses.items()})
            if losses["val"][-1] == min(losses["val"]):
                best_params = params

            elif (
                count_fruitless(losses["val"]) > max_patience
                and (epoch + 1) % check_every == 0
                and epoch > check_every
            ):
                loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
                break
            if jnp.any(jnp.isnan(jnp.array(losses["val"] + losses["train"]))) or jnp.any(
                jnp.isinf(jnp.array(losses["val"] + losses["train"]))
            ):
                loop.set_postfix_str(f"{loop.postfix} (inf or nan loss)")
                break

            if plot_progress and (epoch % int(max_epochs / 10) == 0):
                # plot the data in 2d histograms and the model in 2d histograms
                current_dist = eqx.combine(params, static)
                sample = current_dist.sample(jr.key(123), (100000,))
                if train_data[0].shape[1] == 1:
                    plt.hist(
                        train_data[0],
                        bins=100,
                        histtype="step",
                        label="original",
                        color="blue",
                        density=True,
                    )
                    plt.hist(
                        sample,
                        bins=100,
                        histtype="step",
                        label="resampled",
                        color="red",
                        density=True,
                    )
                    plt.legend()
                else:
                    fig, ax = plt.subplots(
                        train_data[0].shape[1], train_data[0].shape[1], figsize=(10, 10)
                    )
                    for c1 in range(train_data[0].shape[1]):
                        for c2 in range(train_data[0].shape[1]):
                            if c1 <= c2:
                                continue
                            ax[c1, c2].hist2d(*train_data[0][:, [c1, c2]].T, bins=100)
                            ax[c2, c1].hist2d(*sample[:, [c1, c2]].T, bins=100)
                    fig.suptitle("Lower left: data | upper right: model")
                train_summary.figure(f"{plot_progress}", fig, epoch)
                plt.close(fig)

    except KeyboardInterrupt:
        print("Training interrupted by the user")

    params = best_params if return_best else params
    dist = eqx.combine(params, static)
    train_summary.close()
    val_summary.close()
    return dist, losses
