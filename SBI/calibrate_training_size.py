"""
Posterior width and error as a function of the training-set size: for the
posteriors trained on N simulations (models/trained_posterior_adv_<N>_trial20.pkl),
compute the mean posterior std, 90% credible-interval width, MAE and RMSE on
the test set, save them, and plot the normalized std and RMSE against N.

    python SBI/calibrate_training_size.py
"""

# general
import os

# GPU selection, which must happen before jax and torch initialize CUDA
from autocvd import autocvd
autocvd(num_gpus=1, interval=1)

# numerics
import numpy as np

# neural networks
import torch

# plotting
import matplotlib.pyplot as plt

# local
from cwb_sbi.data import load_simulations
from cwb_sbi.paths import (
    CALIBRATION_FIGURES_DIRECTORY,
    DATA_DIRECTORY,
    MODELS_DIRECTORY,
    RESULTS_DIRECTORY,
)
from cwb_sbi.posterior_io import load_posterior


TEST_DATA_PATH = os.path.join(DATA_DIRECTORY, "test_data_960_full.pt")
METRICS_PATH = os.path.join(RESULTS_DIRECTORY, "posterior_std.npz")

TRAINING_SET_SIZES = np.array([100, 500, 1000, 5000, 10000, 20000, 40000])

# Test simulations evaluated and posterior samples per simulation.
NUM_TEST_SIMULATIONS = 500
NUM_POSTERIOR_SAMPLES = 500

# Skip the evaluation and replot the metrics saved in METRICS_PATH.
LOAD_EXISTING_METRICS = False

PARAMETER_LABELS = [r"$\dot M_1$", r"$\dot M_2$", r"$v_{\inf,1}$", r"$v_{\inf,2}$", "e", "i", r"$\eta$"]

plt.rcParams.update({
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


def posterior_metrics(posterior, theta, x):
    """
    Average posterior spread and error of the posterior mean over the first
    NUM_TEST_SIMULATIONS test simulations.

    Returns:
        Dict of per-parameter arrays: mean_posterior_std,
        mean_posterior_ci90_width, mean_mae, mean_rmse.
    """
    num_parameters = theta.shape[1]
    absolute_errors = np.zeros((NUM_TEST_SIMULATIONS, num_parameters))
    root_squared_errors = np.zeros((NUM_TEST_SIMULATIONS, num_parameters))
    posterior_stds = np.zeros((NUM_TEST_SIMULATIONS, num_parameters))
    ci90_widths = np.zeros((NUM_TEST_SIMULATIONS, num_parameters))

    for index in range(NUM_TEST_SIMULATIONS):
        samples = posterior.sample((NUM_POSTERIOR_SAMPLES,), x=x[index]).cpu().numpy()
        true = theta[index].cpu().numpy().copy()

        error = samples.mean(axis=0) - true
        absolute_errors[index] = np.abs(error)
        root_squared_errors[index] = np.sqrt(error**2)

        posterior_stds[index] = np.std(samples, axis=0, ddof=1)
        ci90_widths[index] = np.quantile(samples, 0.95, axis=0) - np.quantile(samples, 0.05, axis=0)

    return {
        'mean_posterior_std': posterior_stds.mean(axis=0),
        'mean_posterior_ci90_width': ci90_widths.mean(axis=0),
        'mean_mae': absolute_errors.mean(axis=0),
        'mean_rmse': root_squared_errors.mean(axis=0),
    }


def plot_metric_versus_training_size(values, ylabel, savefile):
    """Plot a normalized per-parameter metric against the training-set size."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for parameter_index in range(values.shape[1]):
        ax.plot(
            TRAINING_SET_SIZES,
            values[:, parameter_index],
            marker='o',
            label=f'{PARAMETER_LABELS[parameter_index]}',
        )
    ax.set_xscale('log')
    ax.set_xlabel('Training sample size (N)')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.savefig(savefile, dpi=150)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    os.makedirs(CALIBRATION_FIGURES_DIRECTORY, exist_ok=True)

    theta, x = load_simulations(TEST_DATA_PATH)
    theta = theta.to(device)
    x = x.to(device)

    if LOAD_EXISTING_METRICS:
        results = np.load(METRICS_PATH, allow_pickle=True)['results'].item()
    else:
        results = {}
        for training_set_size in TRAINING_SET_SIZES:
            posterior = load_posterior(
                os.path.join(MODELS_DIRECTORY, f"trained_posterior_adv_{training_set_size}_trial20.pkl")
            )
            results[training_set_size] = posterior_metrics(posterior, theta, x)
        np.savez(METRICS_PATH, N_array=TRAINING_SET_SIZES, results=results)

    # Normalize by the typical magnitude of each parameter.
    parameter_scale = np.median(np.abs(theta.cpu().numpy()), axis=0)

    normalized_std = np.array([results[size]['mean_posterior_std'] for size in TRAINING_SET_SIZES]) / parameter_scale
    plot_metric_versus_training_size(
        normalized_std,
        r'Mean posterior $1\sigma$ value (normalized)',
        os.path.join(CALIBRATION_FIGURES_DIRECTORY, "posterior_width_metrics.png"),
    )

    normalized_rmse = np.array([results[size]['mean_rmse'] for size in TRAINING_SET_SIZES]) / parameter_scale
    plot_metric_versus_training_size(
        normalized_rmse,
        'RMSE (normalized)',
        os.path.join(CALIBRATION_FIGURES_DIRECTORY, "posterior_error_per_param.png"),
    )


if __name__ == "__main__":
    main()
