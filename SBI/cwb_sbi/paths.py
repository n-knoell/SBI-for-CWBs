"""
Locations of the data, trained models, results and figures of the SBI
pipeline, so that every script works independently of the working directory.
"""

# general
import os


SBI_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOSITORY_DIRECTORY = os.path.dirname(SBI_DIRECTORY)

# Inputs: simulation banks, test sets, the mass-loss-rate table and the
# reference observations.
DATA_DIRECTORY = os.path.join(SBI_DIRECTORY, "data")
REFERENCE_OBSERVATION_DIRECTORY = os.path.join(DATA_DIRECTORY, "reference_observations")

# Trained (pickled) NPE posteriors.
MODELS_DIRECTORY = os.path.join(SBI_DIRECTORY, "models")

# Non-figure outputs (optuna study database, calibration metrics).
RESULTS_DIRECTORY = os.path.join(SBI_DIRECTORY, "results")
OPTUNA_RESULTS_DIRECTORY = os.path.join(RESULTS_DIRECTORY, "optuna")

FIGURES_DIRECTORY = os.path.join(SBI_DIRECTORY, "figures")
POSTERIOR_FIGURES_DIRECTORY = os.path.join(FIGURES_DIRECTORY, "posteriors")
CALIBRATION_FIGURES_DIRECTORY = os.path.join(FIGURES_DIRECTORY, "calibration")
OPTUNA_FIGURES_DIRECTORY = os.path.join(FIGURES_DIRECTORY, "optuna")
REFERENCE_OBSERVATION_FIGURES_DIRECTORY = os.path.join(FIGURES_DIRECTORY, "reference_observations")

ABC_DIRECTORY = os.path.join(REPOSITORY_DIRECTORY, "ABC")


def reference_observation_path(name):
    """Path of the reference observation ref_obs_<name>.npz."""
    return os.path.join(REFERENCE_OBSERVATION_DIRECTORY, f"ref_obs_{name}.npz")
