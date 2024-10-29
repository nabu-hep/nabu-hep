#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from analysis import Histogram, chi2, chi2_analysis
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data


def plot_history(losses, name):
    plt.plot(losses["train"], label="train")
    plt.plot(losses["val"], label="val")
    plt.legend()
    plt.yscale("log")
    plt.savefig(name)
    plt.close()


def main(args):
    jax.config.update("jax_platform_name", "gpu")

    data = np.load(args.DATAPATH)
    X_train = data["X_train"]
    bins = np.array([chi2.ppf(p, df=5) for p in np.arange(0.0, 1.1, 0.1)])

    while True:

        key, subkey = jr.split(jr.key(np.random.randint(0, high=999999999999)))
        flow = masked_autoregressive_flow(
            subkey,
            base_dist=Normal(jnp.zeros(X_train.shape[1])),
            transformer=None,  # RationalQuadraticSpline(knots=8, interval=4),
            nn_width=args.NNWIDTH,
            nn_depth=args.NNDEPTH,
            flow_layers=args.FLOWLAYERS,
            invert=True,
            nn_activation=jax.nn.relu,
            permutation=list(reversed(range(5))),
        )

        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=args.LR)
        scheduler = optax.exponential_decay(
            init_value=args.LR,
            transition_steps=25,
            decay_rate=0.5,
            staircase=True,
            end_value=args.MINLR,
        )

        # initial
        flow, losses = fit_to_data(
            key,
            flow,
            X_train,
            optimizer=optimizer,
            max_epochs=args.EPOCHS,
            max_patience=50,
            lr_scheduler=scheduler,
        )

        gauss_test = np.array(jax.vmap(flow.bijection.inverse, in_axes=0)(data["X_test"]))

        hist = Histogram(
            dim=5, bins=bins, max_val=20, vals=np.sum(gauss_test**2, axis=1)
        )

        chi2_analysis(
            gauss_test,
            bins=np.array([chi2.ppf(p, df=5) for p in np.arange(0.0, 1, 0.1)] + [20.0]),
            plot_name=os.path.join(args.OUTPATH, "chi2.png"),
        )
        eqx.tree_serialise_leaves(os.path.join(args.OUTPATH, "model.eqx"), flow)
        plot_history(losses, os.path.join(args.OUTPATH, "history.png"))

        pull = hist.pull
        pval = 1.0 - chi2.cdf(np.sum(pull**2), df=len(pull))
        if pval >= 0.03:
            np.savez_compressed(
                os.path.join(args.OUTPATH, "gaussian.npz"), gauss=gauss_test
            )
            break

        print(f"p-value {pval*100:.2f}% Current pull:", hist.pull)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Search model space for proper nflow")

    parameters = parser.add_argument_group("Hyperparameters")
    parameters.add_argument(
        "--nn-width",
        "-w",
        type=int,
        default=32,
        help="Number of nodes per layer",
        dest="NNWIDTH",
    )
    parameters.add_argument(
        "--nn-depth",
        "-d",
        type=int,
        default=3,
        help="Number of layers",
        dest="NNDEPTH",
    )
    parameters.add_argument(
        "--flow-layers",
        "-l",
        type=int,
        default=2,
        help="Number of flow layers",
        dest="FLOWLAYERS",
    )
    parameters.add_argument(
        "-lr", type=float, default=1e-3, help="learning rate", dest="LR"
    )
    parameters.add_argument(
        "-mlr", type=float, default=1e-4, help="minlearning rate", dest="MINLR"
    )
    parameters.add_argument(
        "--epochs", "-e", type=int, default=600, help="Number of epochs", dest="EPOCHS"
    )

    dataset = parser.add_argument_group("Dataset")
    dataset.add_argument(
        "--data-path",
        "-dp",
        type=str,
        help="NumPy file includes standardised dataset. File should include 'X_train` and `X_test` keyword arguments.",
        dest="DATAPATH",
    )

    paths = parser.add_argument_group("Paths")
    paths.add_argument(
        "--out-path",
        "-op",
        type=str,
        help="Output path",
        dest="OUTPATH",
        default="./results",
    )
    paths.add_argument(
        "--out-name",
        "-on",
        type=str,
        help="Output name, default " + datetime.now().strftime("%b%d-%H-%M-%S"),
        dest="OUTNAME",
        default=datetime.now().strftime("%b%d-%H-%M-%S"),
    )

    args = parser.parse_args()

    if not os.path.isdir(args.OUTPATH):
        os.mkdir(args.OUTPATH)

    args.OUTPATH = os.path.join(args.OUTPATH, args.OUTNAME)
    if not os.path.isdir(args.OUTPATH):
        os.mkdir(args.OUTPATH)

    print("<><><> Arguments <><><>")
    for key, item in vars(args).items():
        print(f"   * {key} : {item}")
    print("<><><><><><><><><><><>")

    arg_dict = vars(args)
    arg_dict.update({"HOSTNAME": os.environ.get("HOSTNAME", None)})
    with open(os.path.join(args.OUTPATH, "config.yaml"), "w") as f:
        yaml.safe_dump(arg_dict, f)

    main(args)
