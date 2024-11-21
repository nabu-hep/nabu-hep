from collections.abc import Callable
from collections.abc import Iterator

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flowjax import wrappers
from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import count_fruitless, get_batches, step, train_val_split
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree
from tqdm import tqdm


def batch_split(
    data: np.array, batch_size: int, shuffle: bool = True
) -> Iterator[jnp.array]:
    """Split data into batches

    Args:
        data (np.array): data to be splitted
        batch_size (int): size of each batch
        shuffle (bool, optional): Should the batch be shuffled. Defaults to True.
        number_of_processes (int, optional): If there are multiple availabel device
            this will reshape the batch in chunks to be run in parallel.

    Yields:
        Iterator[jnp.array]: batched data
    """
    indices = np.arange(len(data))
    if shuffle:
        np.random.shuffle(indices)
    batches = np.array_split(indices, len(indices) // batch_size)
    if shuffle:
        np.random.shuffle(batches)
    return (jnp.array(data[batch, :]) for batch in batches)


def fit(
    key: PRNGKeyArray,
    dist: PyTree,  # Custom losses may support broader types than AbstractDistribution
    x: ArrayLike,
    *,
    condition: ArrayLike = None,
    loss_fn: Callable = None,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    check_every: int = 1,
    batch_size: int = 100,
    val_prop: float = 0.1,
    return_best: bool = True,
    show_progress: bool = True,
    lr_scheduler=None,
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

    Returns:
        A tuple containing the trained distribution and the losses.
    """
    data = (x,) if condition is None else (x, condition)
    data = tuple(jnp.asarray(a) for a in data)

    loss_fn = loss_fn or MaximumLikelihoodLoss()

    optimizer = optimizer or optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    )

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

    loop = tqdm(range(max_epochs), disable=not show_progress)

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

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size)):
            key, subkey = jr.split(key)
            loss_i = loss_fn(params, static, *batch, key=subkey)
            batch_losses.append(loss_i)
        losses["val"].append((sum(batch_losses) / len(batch_losses)).item())

        if lr_scheduler is not None:
            opt_state.hyperparams["learning_rate"] = lr_scheduler(epoch + 1)
            losses["lr"].append(float(opt_state.hyperparams["learning_rate"]))

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

    params = best_params if return_best else params
    dist = eqx.combine(params, static)
    return dist, losses
