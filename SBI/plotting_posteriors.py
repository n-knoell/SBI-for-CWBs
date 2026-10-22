"""
Sample the trained NPE posterior for observations and plot the posteriors with
the shared corner-plot routine of the ABC analysis (ABC/abc_inference/plotting.py),
so that NPE and ABC posteriors are directly comparable.

By default the three reference observations in CWB_examples/ are used; set
USE_TEST_SET to plot a simulation of the test set instead. Run from anywhere:

    python SBI/plotting_posteriors.py
"""

# general
import os
import pickle
import sys

# GPU selection, which must happen before jax and torch initialize CUDA
from autocvd import autocvd
autocvd(num_gpus=1)

# jax
import jax

# numerics
import numpy as np

# neural networks
import torch

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# The pickled posterior references the embedding network class by module name.
from helper import DensityEmbedding  # noqa: F401

SBI_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(SBI_DIRECTORY), "ABC"))

# shared posterior plotting (ABC/abc_inference)
from abc_inference.plotting import plot_marginals
from abc_inference.priors import get_prior_ranges
from abc_inference.units import convert_theta_to_phys


# The prior bounds are computed in float64, as in training.
jax.config.update("jax_enable_x64", True)

POSTERIOR_PATH = os.path.join(SBI_DIRECTORY, "trained_posterior_adv_40000_trial20.pkl")
REFERENCE_OBSERVATION_DIRECTORY = os.path.join(SBI_DIRECTORY, "CWB_examples")
OUTPUT_DIRECTORY = os.path.join(SBI_DIRECTORY, "posterior_plots")

REFERENCE_OBSERVATION_NAMES = ["diff", "low", "high"]

# Alternatively plot simulation TEST_NUMBER of the held-out test set.
USE_TEST_SET = False
TEST_DATA_PATH = os.path.join(SBI_DIRECTORY, "sim_data", "test_data_960_full.pt")
TEST_NUMBER = 2

NUM_POSTERIOR_SAMPLES = 50000


def image_grid(images, ncols=6, figsize=None, cmap='viridis', vmin=None, vmax=None, titles=None):
    """
    images: list or array of (H, W) 2D arrays. If images is shape (N, H, W) ok.
    Shows images in a grid.
    """
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[None, ...]
    N = images.shape[0]
    ncols = min(ncols, N)
    nrows = int(np.ceil(N / ncols))
    if figsize is None:
        figsize = (2.2 * ncols, 2.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        ax = axes[i]
        ax.axis('off')
        if i < N:
            norm = LogNorm(vmin=vmin, vmax=vmax)
            im = ax.imshow(images[i], origin='lower', cmap=cmap, norm=norm)
            if titles is not None:
                ax.set_title(titles[i])
    # add colorbar on the right
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cax)
    return fig, axes


def posterior_predictive_grid(posterior_samples, simulator_fn, nplot=9, ncols=3, random_seed=None):
    """
    Posterior predictive: run the simulator (fast) on several posterior samples and show density slices.
    posterior_samples: (N, D) numpy array or torch
    simulator_fn: function(theta) -> 2D array (H, W) or an image
    """
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(posterior_samples), size=min(nplot, len(posterior_samples)), replace=False)
    selected = posterior_samples[idx]
    imgs = []
    for th in selected:
        imgs.append(np.asarray(simulator_fn(th)))
    fig, axes = image_grid(imgs, ncols=ncols)
    return fig, axes


def load_posterior(device):
    """Load the trained NPE posterior and move its network to the device."""
    with open(POSTERIOR_PATH, "rb") as file:
        posterior = pickle.load(file)
    try:
        # Best effort: _net is internal to sbi but commonly present.
        posterior._net.to(device)
    except Exception:
        pass
    return posterior


def load_observations(device):
    """
    Collect the observations to plot.

    Args:
        device: The torch device the observations are moved to.

    Returns:
        A list of (name, x_o, theta_o), with x_o [T, H, W] without the T = 0
        snapshot (the initial condition) and theta_o in code units.
    """
    observations = []
    if USE_TEST_SET:
        test_data = torch.load(TEST_DATA_PATH, map_location="cpu")
        x_o = test_data.get("x")[TEST_NUMBER, 1:, :, :].to(device)
        theta_o = test_data.get("theta")[TEST_NUMBER]
        observations.append((f"full_{TEST_NUMBER}", x_o, theta_o))
    else:
        for name in REFERENCE_OBSERVATION_NAMES:
            data = np.load(
                os.path.join(REFERENCE_OBSERVATION_DIRECTORY, f"ref_obs_{name}.npz"),
                allow_pickle=True,
            )
            x_o = torch.as_tensor(data["one_run"], device=device)[1:, :, :]
            observations.append((f"{name}_full", x_o, data["theta"]))
    return observations


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    posterior = load_posterior(device)
    prior_ranges = get_prior_ranges()

    for name, x_o, theta_o in load_observations(device):
        # The NPE posterior lives in the same parameter space as the ABC
        # samples (natural-log mass-loss rates, code units), so the samples are
        # passed to the shared routine unconverted.
        posterior_samples = posterior.sample((NUM_POSTERIOR_SAMPLES,), x=x_o)
        posterior_samples = posterior_samples.detach().cpu().numpy()

        theta_o = np.asarray(theta_o.cpu() if hasattr(theta_o, "cpu") else theta_o).flatten()

        savefile = os.path.join(OUTPUT_DIRECTORY, f"posterior_{name}.png")
        fig, _ = plot_marginals(
            posterior_samples,
            convert_theta_to_phys(theta_o),
            savefile=savefile,
            prior_ranges=prior_ranges,
        )
        plt.close(fig)
        print(f"Saved {savefile}")


if __name__ == "__main__":
    main()
