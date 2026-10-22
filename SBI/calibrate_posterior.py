"""
Calibration diagnostics of the trained NPE posterior on the held-out test
set: predicted versus true parameters, PP plots, PIT histograms (with
Kolmogorov-Smirnov tests), TARP expected coverage, and SBC rank plots.

    python SBI/calibrate_posterior.py
"""

# general
import os

# GPU selection, which must happen before jax and torch initialize CUDA
from autocvd import autocvd
autocvd(num_gpus=1, interval=1)

# numerics
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kstest

# neural networks
import torch

# plotting
import matplotlib.pyplot as plt

# simulation-based inference
from sbi.analysis.plot import plot_tarp, sbc_rank_plot
from sbi.diagnostics import check_tarp, run_sbc, run_tarp

# units and constants
from astropy import units as u

# local
from cwb_sbi.abc_shared import code_units
from cwb_sbi.data import load_simulations
from cwb_sbi.paths import CALIBRATION_FIGURES_DIRECTORY, DATA_DIRECTORY, MODELS_DIRECTORY
from cwb_sbi.posterior_io import load_posterior


POSTERIOR_PATH = os.path.join(MODELS_DIRECTORY, "trained_posterior_adv_40000_trial20.pkl")
TEST_DATA_PATH = os.path.join(DATA_DIRECTORY, "test_data_960_full.pt")

# Predicted-versus-true plot: test simulations and posterior samples each.
NUM_PREDICTION_SIMULATIONS = 200
FIRST_PREDICTION_INDEX = 0
NUM_PREDICTION_SAMPLES = 2000

# Posterior samples per test simulation for PP/PIT, TARP and SBC.
NUM_PP_SAMPLES = 1000
NUM_TARP_SAMPLES = 1000
NUM_SBC_SAMPLES = 1_000
NUM_PIT_BINS = 10

NUM_PARAMETERS = 7

PREDICTION_TITLES = [
    r"$\log \dot M_1$ [log(M$_{\odot}$/yr)]",
    r"$\log \dot M_2$ [log(M$_{\odot}$/yr)]",
    r"$v_{\infty,1}$ [km/s]",
    r"$v_{\infty,2}$ [km/s]",
    r"Eccentricity $e$",
    r"Inclination $\cos(i)$",
    r"Turbulence parameter $\eta$",
]
PREDICTION_LABELS = [r"$\log \dot M_1$", r"$\log \dot M_2$", r"$v_{\infty,1}$", r"$v_{\infty,2}$", r"$e$", r"$\cos(i)$", r"$\eta$"]
CALIBRATION_LABELS = [r"$\dot M_1$", r"$\dot M_2$", r"$v_{\infty,1}$ ", r"$v_{\infty,2}$", r"$e$", r"$i$", r"$\eta$"]

plt.rcParams.update({
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
})

# Factor converting a mass-loss rate in code units to M_sun / yr.
_MASS_LOSS_RATE_TO_PHYSICAL = (1 * (code_units.code_mass / code_units.code_time)).to(u.M_sun / u.yr).value


def _velocity_to_physical(velocity_code):
    """Wind velocity from code units to km / s."""
    return (velocity_code * code_units.code_velocity).to(u.km / u.s).value


def to_display_units(theta):
    """
    Convert theta in place to display units: log10(Ṁ / (M_sun / yr)) from the
    natural-log code mass-loss rates, and km / s for the wind velocities.

    Args:
        theta: Array of shape (7,) or (N, 7), modified in place.
    """
    for index in (0, 1):
        theta[..., index] = np.log10(_MASS_LOSS_RATE_TO_PHYSICAL) + theta[..., index] * np.log10(np.e)
    for index in (2, 3):
        theta[..., index] = _velocity_to_physical(theta[..., index])


def linear(x, a, b):
    """Linear model a x + b for the fits."""
    return a * x + b


def plot_predicted_versus_true(posterior, theta, x, savefile):
    """
    Plot the posterior mean ± std against the true value for every parameter,
    with an error-weighted linear fit and the identity line.
    """
    true_values = []
    predicted_means = []
    predicted_stds = []
    for index in range(FIRST_PREDICTION_INDEX, FIRST_PREDICTION_INDEX + NUM_PREDICTION_SIMULATIONS):
        samples = posterior.sample((NUM_PREDICTION_SAMPLES,), x=x[index]).cpu().numpy()
        true = theta[index].cpu().numpy().copy()
        to_display_units(true)
        to_display_units(samples)

        true_values.append(true)
        predicted_means.append(np.mean(samples, axis=0))
        predicted_stds.append(np.std(samples, axis=0))

    true_values = np.array(true_values)
    predicted_means = np.array(predicted_means)
    predicted_stds = np.array(predicted_stds)

    fig, axes = plt.subplots(1, NUM_PARAMETERS, figsize=(30, 4))
    for parameter_index, ax in enumerate(axes):
        true_parameter = true_values[:, parameter_index]
        mean_parameter = predicted_means[:, parameter_index]
        std_parameter = predicted_stds[:, parameter_index]

        ax.errorbar(true_parameter, mean_parameter, yerr=std_parameter, fmt='o', ms=3, capsize=2, alpha=0.7)

        (slope, intercept), _ = curve_fit(linear, true_parameter, mean_parameter, sigma=std_parameter, absolute_sigma=True)
        x_fit = np.linspace(np.min(true_parameter), np.max(true_parameter), 200)
        ax.plot(x_fit, linear(x_fit, slope, intercept), linewidth=1.5, label="fit y=ax+b")
        ax.plot(x_fit, x_fit, linestyle="--", linewidth=1.0, label="y=x")

        ax.set_xlabel(f"true {PREDICTION_LABELS[parameter_index]}")
        ax.set_ylabel(f"predicted {PREDICTION_LABELS[parameter_index]}")
        ax.set_title(PREDICTION_TITLES[parameter_index])
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(savefile, dpi=250)
    print(f"Saved {savefile}")


def compute_percentiles(posterior, theta, x):
    """
    Fraction of posterior samples below the true value, per parameter and
    test simulation.

    Returns:
        Array of shape (NUM_PARAMETERS, N).
    """
    num_simulations = x.shape[0]
    percentiles = np.zeros((NUM_PARAMETERS, num_simulations))
    for index in range(num_simulations):
        samples = posterior.sample((NUM_PP_SAMPLES,), x=x[index]).cpu().numpy()
        true = theta[index].cpu().numpy().copy()
        for parameter_index in range(NUM_PARAMETERS):
            percentiles[parameter_index, index] = np.mean(samples[:, parameter_index] <= true[parameter_index])
    return percentiles


def plot_pp(percentiles, savefile):
    """PP plot: sorted percentiles against uniform quantiles."""
    num_simulations = percentiles.shape[1]
    fig, axes = plt.subplots(1, NUM_PARAMETERS, figsize=(28, 4))
    uniform_quantiles = np.linspace(0, 1, num_simulations)
    for parameter_index, ax in enumerate(axes):
        ax.plot(uniform_quantiles, np.sort(percentiles[parameter_index]), marker='o', ms=3, linestyle='-', label='empirical')
        ax.plot([0, 1], [0, 1], linestyle='--', label='ideal')
        ax.set_xlabel('Uniform quantile')
        ax.set_ylabel('Empirical percentile')
        ax.set_title(f'PP-plot for {CALIBRATION_LABELS[parameter_index]}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(savefile, dpi=150)
    plt.clf()
    print(f"Saved {savefile}")

    for parameter_index in range(NUM_PARAMETERS):
        _, p_value = kstest(percentiles[parameter_index, :], 'uniform')
        print(f"param {parameter_index}: KS p-value = {p_value:.3f}")


def plot_pit(percentiles, savefile):
    """PIT histograms of the percentiles, which are flat for calibration."""
    fig, axes = plt.subplots(1, NUM_PARAMETERS, figsize=(4 * NUM_PARAMETERS, 4))
    for parameter_index, ax in enumerate(axes):
        ax.hist(percentiles[parameter_index], bins=NUM_PIT_BINS, range=(0, 1), density=True, alpha=0.75, edgecolor='k')
        ax.axhline(1.0, linestyle='--', linewidth=1, label='uniform density')
        ax.set_xlim(0, 1)
        ax.set_xlabel('PIT value')
        ax.set_title(f'PIT param {parameter_index}')
        ax.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(savefile, dpi=150)
    plt.clf()
    print(f"Saved {savefile}")

    for parameter_index in range(NUM_PARAMETERS):
        _, p_value = kstest(percentiles[parameter_index], 'uniform')
        print(f"param {parameter_index}: PIT KS p-value = {p_value:.3f}")


def plot_tarp_coverage(posterior, theta, x, savefile):
    """
    TARP expected coverage with a 95% binomial (Wald) confidence band.
    """
    expected_coverage, credibility_levels = run_tarp(
        theta,
        x,
        posterior,
        references=None,
        num_posterior_samples=NUM_TARP_SAMPLES,
        # Batched sampling can be faster but may run out of memory.
        use_batched_sampling=False,
    )
    print("ice: ", (expected_coverage - credibility_levels).abs().mean().item())

    expected_coverage = expected_coverage.detach().cpu()
    credibility_levels = credibility_levels.detach().cpu()

    atc, ks_pval = check_tarp(expected_coverage, credibility_levels)
    print("atc:", atc, " (should be close to 0)")
    print("ks_pval:", ks_pval, " (should be > 0.05 ideally)")

    # Standard error of the binomial proportion over the N test simulations.
    num_simulations = int(theta.shape[0])
    coverage = expected_coverage.numpy()
    standard_error = np.sqrt((coverage * (1.0 - coverage)) / float(num_simulations))
    lower = np.clip(coverage - 1.96 * standard_error, 0.0, 1.0)
    upper = np.clip(coverage + 1.96 * standard_error, 0.0, 1.0)

    _, ax = plot_tarp(expected_coverage, credibility_levels)
    ax.fill_between(credibility_levels.numpy(), lower, upper, alpha=0.25, label="95% CI")
    ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(savefile, dpi=150)
    plt.clf()
    print(f"Saved {savefile}")


def plot_sbc(posterior, theta, x, savefile):
    """Simulation-based calibration rank histograms."""
    ranks, _ = run_sbc(
        theta,
        x,
        posterior,
        num_posterior_samples=NUM_SBC_SAMPLES,
        num_workers=1,
        use_batched_sampling=False,
    )
    sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=NUM_SBC_SAMPLES,
        parameter_labels=CALIBRATION_LABELS,
        num_cols=3,
        figsize=(8, 8),
        num_bins=20,
    )
    plt.tight_layout()
    plt.savefig(savefile, dpi=250)
    plt.clf()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CALIBRATION_FIGURES_DIRECTORY, exist_ok=True)

    posterior = load_posterior(POSTERIOR_PATH, device=device)
    theta, x = load_simulations(TEST_DATA_PATH)
    print("x shape: ", x.shape[0])
    theta = theta.to(device)
    x = x.to(device)

    def figure_path(name):
        return os.path.join(CALIBRATION_FIGURES_DIRECTORY, name)

    plot_predicted_versus_true(posterior, theta, x, figure_path(f"params_full{NUM_PREDICTION_SIMULATIONS}_TEST.png"))

    percentiles = compute_percentiles(posterior, theta, x)
    plot_pp(percentiles, figure_path("pp_marginal_posterior_full_c.png"))
    plot_pit(percentiles, figure_path("pit_marginal_posterior_allN_full.png"))

    plot_tarp_coverage(posterior, theta, x, figure_path("tarp_plot_full.png"))
    plot_sbc(posterior, theta, x, figure_path("SBC_plot_full.png"))


if __name__ == "__main__":
    main()
