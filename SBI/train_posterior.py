"""
Train the NPE posterior on the clean simulations with the tuned architecture
(optuna trial 20), resampling the detector noise every few epochs so that the
network never sees the same noise realisation for long.

Optionally trains one posterior per training-set size (used by
calibrate_training_size.py).

    python SBI/train_posterior.py
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

# simulation-based inference
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# local
from cwb_sbi.data import load_simulations
from cwb_sbi.embedding_nets import make_temporal_embedding
from cwb_sbi.noise import add_fresh_noise
from cwb_sbi.paths import DATA_DIRECTORY, MODELS_DIRECTORY
from cwb_sbi.posterior_io import save_posterior
from cwb_sbi.priors import make_prior


SIMULATIONS_PATH = os.path.join(DATA_DIRECTORY, "simulations_clean_74.pt")

NUM_SIMULATIONS = 20

# Noise is resampled every EPOCHS_PER_NOISE_ROUND epochs, for at most
# MAX_NOISE_ROUNDS rounds, with sbi's early stopping after
# STOP_AFTER_EPOCHS epochs without validation-loss improvement.
EPOCHS_PER_NOISE_ROUND = 5
MAX_NOISE_ROUNDS = 60
STOP_AFTER_EPOCHS = 20

# Alternatively train one posterior per training-set size, each with a single
# noise realisation.
TRAIN_FOR_TRAINING_SET_SIZES = False
TRAINING_SET_SIZES = [20000]

# Architecture of optuna trial 20 (best TARP). Trial 14 (best combined) was
# num_conv_blocks=3, fc_dim=64, base_channels=16, temporal_pool_output=1,
# temporal_num_layers=1, temporal_stride_first=2, hidden_features=30,
# num_transforms=20.
EMBEDDING_HYPERPARAMETERS = {
    "num_conv_blocks": 3,
    "fc_dim": 128,
    "base_channels": 32,
    "temporal_pool_output": 2,
    "temporal_num_layers": 2,
    "temporal_stride_first": 1,
}
HIDDEN_FEATURES = 7
NUM_TRANSFORMS = 20


def random_seed():
    """A random 31-bit seed for the noise realisation."""
    return int(np.random.randint(0, 2**31 - 1))


def build_density_estimator():
    """Neural spline flow with the tuned temporal embedding network."""
    return posterior_nn(
        model="nsf",
        hidden_features=HIDDEN_FEATURES,
        num_transforms=NUM_TRANSFORMS,
        embedding_net=make_temporal_embedding(**EMBEDDING_HYPERPARAMETERS),
    )


def train_with_noise_resampling(theta, x_clean, prior, density_estimator_builder, device):
    """
    Train on NUM_SIMULATIONS random simulations, swapping in a fresh noise
    realisation of the same simulations every noise round.

    Args:
        theta: All parameters (on device).
        x_clean: All clean observations (on CPU).
        prior: The prior.
        density_estimator_builder: The sbi density estimator builder.
        device: The training device.

    Returns:
        The trained posterior.
    """
    print(f"Training with {NUM_SIMULATIONS} simulations, "
          f"resampling noise every {EPOCHS_PER_NOISE_ROUND} epochs "
          f"(cap: {MAX_NOISE_ROUNDS * EPOCHS_PER_NOISE_ROUND} total epochs)...")

    inference = NPE(prior=prior, density_estimator=density_estimator_builder, device=device)
    indices = torch.randperm(theta.shape[0])[:NUM_SIMULATIONS]
    theta_sample = theta[indices]

    # The clean simulations stay on the CPU; only the noisy batch is moved to
    # the device.
    x_sample_clean = x_clean[indices.cpu()]

    initial_seed = random_seed()
    print(f"[round 0] initial noise seed: {initial_seed}")
    x_sample_noisy = add_fresh_noise(x_sample_clean, seed=initial_seed).to(device)
    inference.append_simulations(theta_sample, x_sample_noisy)

    density_estimator = None
    for round_index in range(MAX_NOISE_ROUNDS):
        if round_index > 0:
            # NOTE: this swaps the data in sbi's internal round-wise buffer
            # (`_x_roundwise`), whose name has been stable across sbi
            # 0.22-0.25; adjust here if a newer sbi renames it.
            seed = random_seed()
            x_new = add_fresh_noise(x_sample_clean, seed=seed).to(device)
            inference._x_roundwise[-1] = x_new
            del x_new
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[round {round_index}] noise seed: {seed}")

        # max_num_epochs is an absolute cap in sbi, so it grows every round.
        density_estimator = inference.train(
            max_num_epochs=(round_index + 1) * EPOCHS_PER_NOISE_ROUND,
            stop_after_epochs=STOP_AFTER_EPOCHS,
            resume_training=(round_index > 0),
        )

        epochs_without_improvement = getattr(inference, "_epochs_since_last_improvement", 0)
        if epochs_without_improvement >= STOP_AFTER_EPOCHS:
            print(f"  -> converged after round {round_index + 1} "
                  f"({epochs_without_improvement} epochs without val-loss improvement). Stopping.")
            break
    else:
        print(f"  -> hit max_noise_rounds={MAX_NOISE_ROUNDS} without convergence.")

    return inference.build_posterior(density_estimator)


def train_for_training_set_sizes(theta, x_clean, prior, density_estimator_builder, device):
    """Train and save one posterior per entry of TRAINING_SET_SIZES."""
    for num_simulations in TRAINING_SET_SIZES:
        print(f"Training with {num_simulations} simulations...")
        inference = NPE(prior=prior, density_estimator=density_estimator_builder, device=device)
        indices = torch.randperm(theta.shape[0])[:num_simulations]
        theta_sample = theta[indices]

        # A new seed per run, so that every retrain sees different detector
        # noise on top of the same clean simulations.
        noise_seed = random_seed()
        print(f"  noise seed: {noise_seed}")
        x_sample = add_fresh_noise(x_clean[indices.cpu()], seed=noise_seed).to(device)

        density_estimator = inference.append_simulations(theta_sample, x_sample).train()
        posterior = inference.build_posterior(density_estimator)
        path = os.path.join(MODELS_DIRECTORY, f"trained_posterior_adv_{num_simulations}_TEST.pkl")
        save_posterior(posterior, path)
        print(f"Saved posterior -> {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODELS_DIRECTORY, exist_ok=True)

    theta, x_clean = load_simulations(SIMULATIONS_PATH)
    theta = theta.to(device)
    print("x_clean.shape:", x_clean.shape)
    print("theta.shape:", theta.shape)

    prior = make_prior(device=device)
    density_estimator_builder = build_density_estimator()

    if TRAIN_FOR_TRAINING_SET_SIZES:
        train_for_training_set_sizes(theta, x_clean, prior, density_estimator_builder, device)
        return

    posterior = train_with_noise_resampling(theta, x_clean, prior, density_estimator_builder, device)
    path = os.path.join(MODELS_DIRECTORY, f"trained_posterior_adv_{NUM_SIMULATIONS}_TEST.pkl")
    save_posterior(posterior, path)
    print(f"Saved posterior -> {path}")


if __name__ == "__main__":
    main()
