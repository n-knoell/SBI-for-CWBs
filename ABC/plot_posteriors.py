"""
Re-plot the saved ABC rejection posteriors (results/abc_rejection_<name>.npz)
against the true parameters of the matching reference observations, without
re-running the sampler.

Run from anywhere:

    python ABC/plot_posteriors.py
"""

# general
import os

# GPU selection, which must happen before jax initializes CUDA
from autocvd import autocvd
autocvd(num_gpus=1)

# jax
import jax

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# local
from abc_inference.plotting import plot_marginals
from abc_inference.priors import get_prior_ranges
from abc_inference.units import convert_theta_to_phys


# The prior bounds are computed in float64, as in the sampling run.
jax.config.update("jax_enable_x64", True)

OBSERVATION_NAMES = ["diff", "low", "high"]

ABC_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_DIRECTORY = os.path.dirname(ABC_DIRECTORY)
RESULTS_DIRECTORY = os.path.join(ABC_DIRECTORY, "results")
FIGURES_DIRECTORY = os.path.join(ABC_DIRECTORY, "figures")
REFERENCE_OBSERVATION_DIRECTORY = os.path.join(REPOSITORY_DIRECTORY, "SBI", "data", "reference_observations")


def main():
    os.makedirs(FIGURES_DIRECTORY, exist_ok=True)
    prior_ranges = get_prior_ranges()

    for name in OBSERVATION_NAMES:
        posterior = np.load(os.path.join(RESULTS_DIRECTORY, f"abc_rejection_{name}.npz"))["posterior"]
        true_theta = np.load(
            os.path.join(REFERENCE_OBSERVATION_DIRECTORY, f"ref_obs_{name}.npz"),
            allow_pickle=True,
        )["theta"]

        savefile = os.path.join(FIGURES_DIRECTORY, f"abc_rejection_{name}.png")
        fig, _ = plot_marginals(
            posterior,
            convert_theta_to_phys(true_theta.flatten()),
            savefile=savefile,
            prior_ranges=prior_ranges,
        )
        plt.close(fig)
        print(f"Saved {savefile} ({posterior.shape[0]} samples)")


if __name__ == "__main__":
    main()
