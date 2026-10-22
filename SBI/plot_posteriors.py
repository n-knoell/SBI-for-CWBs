"""
Sample the trained NPE posterior for observations and plot the posteriors with
the corner-plot routine shared with the ABC analysis, so that NPE and ABC
posteriors are directly comparable.

By default the three reference observations are used; set USE_TEST_SET to
plot a simulation of the test set instead.

    python SBI/plot_posteriors.py
"""

# general
import os

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

# local
from cwb_sbi.abc_shared import convert_theta_to_phys, get_prior_ranges, plot_marginals
from cwb_sbi.paths import (
    DATA_DIRECTORY,
    MODELS_DIRECTORY,
    POSTERIOR_FIGURES_DIRECTORY,
    reference_observation_path,
)
from cwb_sbi.posterior_io import load_posterior


# The prior bounds (plot limits) are computed in float64.
jax.config.update("jax_enable_x64", True)

POSTERIOR_PATH = os.path.join(MODELS_DIRECTORY, "trained_posterior_adv_40000_trial20.pkl")

REFERENCE_OBSERVATION_NAMES = ["diff", "low", "high"]

# Alternatively plot simulation TEST_NUMBER of the held-out test set.
USE_TEST_SET = False
TEST_DATA_PATH = os.path.join(DATA_DIRECTORY, "test_data_960_full.pt")
TEST_NUMBER = 2

NUM_POSTERIOR_SAMPLES = 50000


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
            data = np.load(reference_observation_path(name), allow_pickle=True)
            x_o = torch.as_tensor(data["one_run"], device=device)[1:, :, :]
            observations.append((f"{name}_full", x_o, data["theta"]))
    return observations


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(POSTERIOR_FIGURES_DIRECTORY, exist_ok=True)

    posterior = load_posterior(POSTERIOR_PATH, device=device)
    prior_ranges = get_prior_ranges()

    for name, x_o, theta_o in load_observations(device):
        # The NPE posterior lives in the same parameter space as the ABC
        # samples (natural-log mass-loss rates, code units), so the samples are
        # passed to the shared routine unconverted.
        posterior_samples = posterior.sample((NUM_POSTERIOR_SAMPLES,), x=x_o)
        posterior_samples = posterior_samples.detach().cpu().numpy()

        theta_o = np.asarray(theta_o.cpu() if hasattr(theta_o, "cpu") else theta_o).flatten()

        savefile = os.path.join(POSTERIOR_FIGURES_DIRECTORY, f"posterior_{name}.png")
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
