"""
Plot the results of the optuna hyperparameter study: NLL versus mean TARP
deviation of all completed trials with the chosen trials highlighted, and the
Pareto front. Also prints trial 20 and the trial with the best TARP value.

    python SBI/plot_optuna_study.py
"""

# general
import os

# GPU selection
from autocvd import autocvd
autocvd(num_gpus=1)

# plotting
import matplotlib.pyplot as plt

# hyperparameter optimisation
import optuna

# local
from cwb_sbi.paths import OPTUNA_FIGURES_DIRECTORY, OPTUNA_RESULTS_DIRECTORY


STUDY_NAME = "study_full_adv2"
STORAGE = f"sqlite:///{os.path.join(OPTUNA_RESULTS_DIRECTORY, STUDY_NAME + '.db')}"

# Positions in the list of completed trials. Trial 20 has the best TARP value
# and trial 14 the best combined value; both were used for training.
HIGHLIGHTED_TRIALS = [0, 20, 14, 98, 124]

plt.rcParams.update({
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


def print_trial_summary(completed_trials):
    """Print trial 20 and the trial with the smallest TARP deviation."""
    print(completed_trials[20])

    trials_with_values = [trial for trial in completed_trials if trial.values]
    if not trials_with_values:
        print("No completed trials with objective values.")
        return
    best = min(trials_with_values, key=lambda trial: trial.values[1])
    print("trial.number:", best.number)
    print("values:", best.values)
    print("params:", best.params)


def plot_nll_versus_tarp(completed_trials, savefile):
    """Scatter the two objectives of all completed trials."""
    objective_values = [trial.values for trial in completed_trials if trial.values is not None]
    nll = [values[0] for values in objective_values]
    tarp_deviation = [values[1] for values in objective_values]

    plt.figure(figsize=(8, 6))
    plt.scatter(nll, tarp_deviation, c='blue', label='Other trials')

    for trial_index in HIGHLIGHTED_TRIALS:
        if trial_index < len(nll):
            plt.scatter(nll[trial_index], tarp_deviation[trial_index], c='red', s=40)
    plt.scatter(nll[20], tarp_deviation[20], c='red', s=40, label='Best trials')

    plt.scatter(nll[20], tarp_deviation[20], c='green', s=60, zorder=5)
    plt.scatter(nll[14], tarp_deviation[14], c='green', s=60, zorder=5, label='Chosen trials')
    for trial_index in [14, 20]:
        plt.annotate(
            str(trial_index),
            (nll[trial_index], tarp_deviation[trial_index]),
            fontsize=10,
            alpha=0.9,
            xytext=(-18, 0),
            textcoords="offset points",
        )

    plt.ylabel('mean TARP deviation')
    plt.xlabel('NLL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savefile)
    plt.clf()


def main():
    os.makedirs(OPTUNA_FIGURES_DIRECTORY, exist_ok=True)
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

    print_trial_summary(completed_trials)
    plot_nll_versus_tarp(completed_trials, os.path.join(OPTUNA_FIGURES_DIRECTORY, "optuna_NLLvsTARP_adv.png"))

    optuna.visualization.matplotlib.plot_pareto_front(study, target_names=[r"NLL", r"mean TARP dev"])
    plt.savefig(os.path.join(OPTUNA_FIGURES_DIRECTORY, "optuna_study_adv.png"))


if __name__ == "__main__":
    main()
