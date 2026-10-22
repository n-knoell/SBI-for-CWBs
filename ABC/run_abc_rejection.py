"""
Run ABC rejection sampling for one reference colliding-wind binary Hα
observation against a bank of precomputed simulations, then save the posterior,
plot its marginals and print the 68% credible-interval widths.

Run from the repository root:

    python ABC/run_abc_rejection.py
"""

# general
import os
from functools import partial

# GPU selection, which must happen before jax and torch initialize CUDA
from autocvd import autocvd
autocvd(num_gpus=1)

# jax
import jax

# numerics
import numpy as np

# neural networks
import torch

# local
from abc_inference.diagnostics import report_widths
from abc_inference.distances import standardized_l2
from abc_inference.plotting import plot_marginals
from abc_inference.priors import get_prior_ranges
from abc_inference.rejection import abc_rejection
from abc_inference.summaries import compute_summary_norm, physical_summary
from abc_inference.units import convert_theta_to_phys


# The summary statistics are computed in float64.
jax.config.update("jax_enable_x64", True)

# Which reference observation to fit (one of "diff", "low", "high").
OBSERVATION_NAME = "diff"

# Input data, relative to the working directory.
REFERENCE_OBSERVATION_PATH = f"ref_obs_{OBSERVATION_NAME}.npz"
SIMULATION_BANK_PATH = "sim_data/full_comb_40k.pt"

# Outputs are written next to this script.
ABC_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIRECTORY = os.path.join(ABC_DIRECTORY, "results")
FIGURES_DIRECTORY = os.path.join(ABC_DIRECTORY, "figures")

def main():
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    os.makedirs(FIGURES_DIRECTORY, exist_ok=True)

    # -------------------------------------------------------------
    # =============== ↓ Reference observation ↓ ===================
    # -------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_data = np.load(REFERENCE_OBSERVATION_PATH, allow_pickle=True)
    observed_run = torch.as_tensor(reference_data["one_run"], device=device)
    true_theta = torch.as_tensor(reference_data["theta"], device=device)

    # Drop the T = 0 snapshot, which is just the initial condition.
    observed_images = observed_run[1:, :, :]

    true_values = true_theta.cpu().numpy().flatten()
    true_values_phys = convert_theta_to_phys(true_values)
    print("True values (physical units): ", true_values_phys)

    observed_summary = physical_summary(observed_images)

    # -------------------------------------------------------------
    # =============== ↑ Reference observation ↑ ===================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================ ↓ Summary normalization ↓ ==================
    # -------------------------------------------------------------

    simulations = torch.load(SIMULATION_BANK_PATH, map_location="cpu")
    simulated_images = simulations.get("x")[:, 1:, :, :].numpy()

    summary_mean, summary_std = compute_summary_norm(
        simulated_images,
        physical_summary,
        max_sims=1000,
    )
    distance_fn = partial(standardized_l2, mean=summary_mean, std=summary_std)

    # -------------------------------------------------------------
    # ================ ↑ Summary normalization ↑ ==================
    # -------------------------------------------------------------

    prior_ranges = get_prior_ranges()

    posterior, distances, epsilon = abc_rejection(
        observed_summary=observed_summary,
        prior_ranges=prior_ranges,
        simulations=simulations,
        num_samples=2000,
        batch_size=4,
        rng_seed=1,
        summary_fn=physical_summary,
        distance_fn=distance_fn,
        save_path=os.path.join(RESULTS_DIRECTORY, f"abc_rejection_{OBSERVATION_NAME}.npz"),
    )
    print("ABC-Rejection sample shape:", posterior.shape)

    np.savez_compressed(os.path.join(RESULTS_DIRECTORY, f"posterior_{OBSERVATION_NAME}.npz"), arr=posterior)
    np.savez_compressed(os.path.join(RESULTS_DIRECTORY, f"dists_{OBSERVATION_NAME}.npz"), arr=distances)
    np.savez_compressed(os.path.join(RESULTS_DIRECTORY, f"eps_{OBSERVATION_NAME}.npz"), arr=epsilon)

    plot_marginals(
        posterior,
        true_values_phys,
        savefile=os.path.join(FIGURES_DIRECTORY, f"abc_rejection_{OBSERVATION_NAME}.png"),
        prior_ranges=prior_ranges,
    )

    print(report_widths(posterior, prior_ranges, name=OBSERVATION_NAME))


if __name__ == "__main__":
    main()
