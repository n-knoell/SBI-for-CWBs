"""
Generate the training simulations: draw parameters from the prior, simulate
the clean Hα photon-count time series in chunks and save them to
data/simulations_clean.pt. Detector noise is added later, at training time.

Optionally (re)creates the three reference observations (low / diff / high)
and plots them.

    python SBI/run_simulations.py
"""

# general
import os
from timeit import default_timer as timer

# GPU selection, which must happen before jax and torch initialize CUDA
from autocvd import autocvd
autocvd(num_gpus=1, interval=1)

# jax
import jax
import jax.numpy as jnp
from jax import random

# numerics
import numpy as np

# neural networks
import torch

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# units and constants
from astropy import units as u

# local
from cwb_sbi.abc_shared import code_units
from cwb_sbi.paths import (
    DATA_DIRECTORY,
    REFERENCE_OBSERVATION_DIRECTORY,
    REFERENCE_OBSERVATION_FIGURES_DIRECTORY,
    reference_observation_path,
)
from cwb_sbi.priors import make_prior
from cwb_sbi.simulator import sample_simulation


NUM_SIMULATIONS = 4500
# Number of simulations run in parallel (vmapped) per batch.
CHUNK_SIZE = 150
OUTPUT_PATH = os.path.join(DATA_DIRECTORY, "simulations_clean.pt")

RUN_SIMULATION_BANK = True
MAKE_REFERENCE_OBSERVATIONS = False
PLOT_REFERENCE_OBSERVATIONS = False

# Physical parameters of the reference observations: Ṁ_1, Ṁ_2 [M_sun / yr],
# v_inf_1, v_inf_2 [km / s], e, cos(i), η.
REFERENCE_PARAMETERS = {
    "low": (6.5e-8, 1.5e-8, 2010, 1680, 0.21, 0.35, 0.011),
    "diff": (8.4e-6, 1.2e-8, 3080, 1740, 0.09, 0.89, 0.007),
    "high": (4e-6, 4e-6, 2977, 2505, 0.82, 0.62, 0.016),
}


def reference_theta(mlr1, mlr2, wind_velocity1, wind_velocity2, eccentricity, cos_inclination, turbulence_strength):
    """
    Convert physical reference parameters to the theta vector in code units.

    Returns:
        torch tensor (log Ṁ_1, log Ṁ_2, v_inf_1, v_inf_2, e, cos(i), η).
    """
    code_mass_loss_rate_unit = code_units.code_mass / code_units.code_time
    log_mlr1 = float(jnp.log((mlr1 * u.M_sun / u.yr).to(code_mass_loss_rate_unit).value))
    log_mlr2 = float(jnp.log((mlr2 * u.M_sun / u.yr).to(code_mass_loss_rate_unit).value))
    wind_velocity1 = (wind_velocity1 * u.km / u.s).to(code_units.code_velocity).value
    wind_velocity2 = (wind_velocity2 * u.km / u.s).to(code_units.code_velocity).value
    return torch.tensor([
        log_mlr1,
        log_mlr2,
        wind_velocity1,
        wind_velocity2,
        eccentricity,
        cos_inclination,
        turbulence_strength,
    ])


def run_simulation_bank():
    """Simulate NUM_SIMULATIONS prior draws in chunks and save them."""
    batched_simulator = jax.vmap(sample_simulation)
    prior = make_prior()
    theta = jnp.asarray(prior.sample((NUM_SIMULATIONS,)).numpy())

    start_time = timer()
    simulated_chunks = []
    used_thetas = []
    for start in range(0, theta.shape[0], CHUNK_SIZE):
        # Every chunk gets fresh keys from a random seed.
        chunk_key = jax.random.PRNGKey(np.random.randint(0, 1e6))
        keys = jax.random.split(chunk_key, num=CHUNK_SIZE)
        theta_chunk = theta[start:start + CHUNK_SIZE]
        simulated_chunks.append(batched_simulator(theta_chunk, keys))
        used_thetas.append(theta_chunk)

    x = torch.from_numpy(np.asarray(jnp.concatenate(simulated_chunks, axis=0)).copy())
    theta_used = torch.from_numpy(np.asarray(jnp.concatenate(used_thetas, axis=0)).copy())

    torch.save({"theta": theta_used, "x": x}, OUTPUT_PATH)
    print(f"Saved simulations (clean photon counts) to {OUTPUT_PATH}")
    print(f"Time taken for {NUM_SIMULATIONS} simulations: {timer() - start_time:.2f} seconds")


def plot_reference_observation(one_run, name):
    """Plot the ten post-initial snapshots of an observation in one figure."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    num_plots = min(one_run.shape[0], axes.size)
    axes = axes.reshape(-1)
    for index in range(num_plots):
        ax = axes[index]
        image = ax.imshow(np.asarray(one_run[index + 1]), cmap='inferno', origin='lower', norm=LogNorm())
        ax.set_title(f"T = {(index + 1) * 0.5} yrs", fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    colorbar = fig.colorbar(image, ax=axes, orientation='vertical', shrink=1.0, aspect=12, pad=0.02)
    colorbar.set_label('Counts', fontsize=14)
    plt.savefig(os.path.join(REFERENCE_OBSERVATION_FIGURES_DIRECTORY, f"ref_{name}.png"))
    plt.close()


def make_reference_observation(theta, name):
    """Simulate, save and plot one reference observation (PRNG key 0)."""
    theta_jax = jnp.asarray(theta.numpy())
    one_run = sample_simulation(theta_jax, random.PRNGKey(0))
    np.savez(reference_observation_path(name), theta=theta_jax, one_run=np.asarray(one_run))
    plot_reference_observation(one_run, name)


def main():
    if RUN_SIMULATION_BANK:
        run_simulation_bank()

    if MAKE_REFERENCE_OBSERVATIONS:
        os.makedirs(REFERENCE_OBSERVATION_DIRECTORY, exist_ok=True)
        for name, parameters in REFERENCE_PARAMETERS.items():
            make_reference_observation(reference_theta(*parameters), name)

    if PLOT_REFERENCE_OBSERVATIONS:
        for name in REFERENCE_PARAMETERS:
            data = np.load(reference_observation_path(name), allow_pickle=True)
            plot_reference_observation(data["one_run"], name)


if __name__ == "__main__":
    main()
