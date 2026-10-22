"""
Multi-objective optuna search over the embedding and flow hyperparameters,
minimising the test-set negative log-likelihood and the mean TARP coverage
deviation. The study is stored in results/optuna/study_full_adv2.db and can
be resumed.

    python SBI/optimize_hyperparameters.py
"""

# general
import os
from functools import partial

# GPU selection, which must happen before jax and torch initialize CUDA
from autocvd import autocvd
autocvd(num_gpus=1, interval=1)

# neural networks
import torch

# simulation-based inference
from sbi.diagnostics import run_tarp
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# hyperparameter optimisation
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

# local
from cwb_sbi.data import load_simulations
from cwb_sbi.diagnostics import compute_log_prob_safe
from cwb_sbi.embedding_nets import make_temporal_embedding
from cwb_sbi.paths import DATA_DIRECTORY, OPTUNA_RESULTS_DIRECTORY
from cwb_sbi.priors import make_prior


TRAINING_DATA_PATH = os.path.join(DATA_DIRECTORY, "full_comb.pt")
TEST_DATA_PATH = os.path.join(DATA_DIRECTORY, "test_data_full.pt")
NUM_TRAINING_SIMULATIONS = 25000

STUDY_NAME = "study_full_adv2"
STORAGE = f"sqlite:///{os.path.join(OPTUNA_RESULTS_DIRECTORY, STUDY_NAME + '.db')}"
MAX_TRIALS = 128

# Test simulations and posterior samples per simulation used for TARP.
NUM_TARP_SIMULATIONS = 500
NUM_TARP_POSTERIOR_SAMPLES = 500


def make_embedding_from_trial_temporal(trial, input_shape=(10, 64, 64)):
    """
    Temporal embedding with architecture hyperparameters sampled from the
    trial.

    Args:
        trial: The optuna trial.
        input_shape: (T, H, W) of the observations.

    Returns:
        A DensityEmbeddingPerFrameTemporalAdv.
    """
    return make_temporal_embedding(
        num_conv_blocks=trial.suggest_categorical("num_conv_blocks", [2, 3]),
        fc_dim=trial.suggest_categorical("fc_dim", [64, 128, 256]),
        base_channels=trial.suggest_categorical("base_channels", [16, 32]),
        temporal_pool_output=trial.suggest_categorical("temporal_pool_output", [1, 2, 4]),
        temporal_num_layers=trial.suggest_int("temporal_num_layers", 1, 3),
        temporal_stride_first=trial.suggest_categorical("temporal_stride_first", [1, 2]),
        input_shape=input_shape,
    )


def objective(trial, prior, theta, x, theta_test, x_test):
    """
    Train an NPE posterior with the trial's hyperparameters and evaluate it.

    Args:
        trial: The optuna trial.
        prior: The prior.
        theta, x: Training data (on device).
        theta_test, x_test: Test data (on device).

    Returns:
        (test-set NLL, mean absolute TARP coverage deviation).
    """
    embedding_net = make_embedding_from_trial_temporal(trial)
    hidden_features = trial.suggest_int("hidden_features", 4, 30, log=False)
    num_transforms = trial.suggest_int("num_transforms", 3, 25, log=False)

    density_estimator_builder = posterior_nn(
        model="nsf",
        embedding_net=embedding_net,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
    )
    inference = NPE(prior=prior, density_estimator=density_estimator_builder, show_progress_bars=True, device="cuda")
    density_estimator = inference.append_simulations(theta, x).train(
        show_train_summary=True,
        training_batch_size=5000,
    )
    posterior = inference.build_posterior(density_estimator)

    log_probs = compute_log_prob_safe(posterior, theta_test, x_test, chunk_size=80)
    test_nll = -log_probs.mean()

    expected_coverage, credibility_levels = run_tarp(
        theta_test[:NUM_TARP_SIMULATIONS],
        x_test[:NUM_TARP_SIMULATIONS],
        posterior,
        num_posterior_samples=NUM_TARP_POSTERIOR_SAMPLES,
        use_batched_sampling=True,
    )
    mean_coverage_deviation = (expected_coverage - credibility_levels).abs().mean().item()

    return test_nll, mean_coverage_deviation


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    theta, x = load_simulations(TRAINING_DATA_PATH, map_location=None)
    theta = theta[:NUM_TRAINING_SIMULATIONS].to(device)
    x = x[:NUM_TRAINING_SIMULATIONS].to(device)
    print("x.shape:", x.shape)
    print("theta.shape:", theta.shape)

    theta_test, x_test = load_simulations(TEST_DATA_PATH, map_location=None)
    theta_test = theta_test.to(device)
    x_test = x_test.to(device)
    print("theta_test.shape:", theta_test.shape)
    print("x_test.shape:", x_test.shape)

    prior = make_prior(device=device)

    os.makedirs(OPTUNA_RESULTS_DIRECTORY, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )
    study.optimize(
        partial(objective, prior=prior, theta=theta, x=x, theta_test=theta_test, x_test=x_test),
        callbacks=[MaxTrialsCallback(MAX_TRIALS, states=(TrialState.COMPLETE, TrialState.FAIL))],
    )


if __name__ == "__main__":
    main()
