"""
ABC rejection sampling, either against a bank of precomputed simulations or by
calling a simulator on fresh prior draws.
"""

# general
from time import time

# jax
import jax

# numerics
import numpy as np

# local
from abc_inference.distances import standardized_l2
from abc_inference.summaries import physical_summary


def _to_numpy(array):
    """Convert a torch tensor (on any device) or array-like to numpy."""
    return array.cpu().numpy() if hasattr(array, "cpu") else np.array(array)


def abc_rejection(
    observed_summary,
    prior_ranges,
    simulations=None,
    num_samples=100,
    epsilon=None,
    batch_size=4,
    rng_seed=0,
    summary_fn=physical_summary,
    distance_fn=standardized_l2,
    simulator_fn=None,
    save_path="ABC/abc_rejection.npz",
):
    """
    Run ABC rejection sampling and save the accepted samples.

    Proposals whose summary lies within epsilon of the observed summary are
    accepted. With precomputed simulations the proposals are the bank's thetas
    in order, and sampling stops early if the bank is exhausted.

    Args:
        observed_summary: Summary vector of the observation.
        prior_ranges: List of (low, high) per parameter, in the units the
            simulator expects.
        simulations: Optional dict with "theta" [N, D] and "x" [N, T+1, H, W]
            (torch tensors or arrays). The T = 0 snapshot is dropped.
        num_samples: Number of accepted posterior samples to collect.
        epsilon: Acceptance threshold. If None, it is set to the 5% quantile
            of the distances of up to 500 pilot simulations.
        batch_size: Proposals per loop iteration (amortizes loop overhead).
        rng_seed: Seed of the jax PRNG used for prior draws.
        summary_fn: Function mapping an observation to its summary vector.
        distance_fn: Function (summ_sim, summ_obs) -> scalar distance.
        simulator_fn: Simulator called as simulator_fn(theta, summary_fn=...)
            when no precomputed simulations are given.
        save_path: Path of the .npz file the results are written to.

    Returns:
        (accepted thetas [n_accepted, D], their distances, epsilon).
    """
    key = jax.random.PRNGKey(rng_seed)
    accepted = []
    accepted_distances = []

    # NOTE: sample_prior_uniform is not defined anywhere in this package, so
    # the simulator_fn code path (simulations=None) currently raises a
    # NameError. Only the precomputed-simulations path is in use.

    if simulations is not None:
        # Drop the T = 0 snapshot, which is just the initial condition.
        all_thetas = _to_numpy(simulations.get("theta"))
        all_sims = _to_numpy(simulations.get("x")[:, 1:, :, :])
        num_precomputed = all_thetas.shape[0]
    else:
        num_precomputed = 0

    def simulate_summary(theta, simulation_index):
        if simulations is None:
            simulation = simulator_fn(theta, summary_fn=summary_fn)
        else:
            simulation = all_sims[simulation_index]
        return summary_fn(simulation)

    # -------------------------------------------------------------
    # ============ ↓ Pilot run to set the threshold ↓ =============
    # -------------------------------------------------------------

    if epsilon is None:
        pilot_size = 500
        pilot_quantile = 0.05
        subkey, key = jax.random.split(key)
        if simulations is None:
            pilot_thetas = np.array(sample_prior_uniform(subkey, pilot_size, prior_ranges))
        else:
            pilot_thetas = all_thetas[:min(pilot_size, num_precomputed)]

        pilot_distances = [
            float(distance_fn(simulate_summary(theta, index), observed_summary))
            for index, theta in enumerate(pilot_thetas)
        ]
        epsilon = np.quantile(np.array(pilot_distances), pilot_quantile)
        print(f"[ABC] Pilot epsilon set to {epsilon:.3e} ({pilot_quantile*100:.2f}% quantile)")

    # -------------------------------------------------------------
    # ============ ↑ Pilot run to set the threshold ↑ =============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =================== ↓ Rejection loop ↓ ======================
    # -------------------------------------------------------------

    num_loops = 0
    start_time = time()

    # Pointer into the precomputed arrays when simulations are given.
    simulation_index = 0

    while len(accepted) < num_samples:
        num_loops += 1
        subkey, key = jax.random.split(key)

        if simulations is None:
            thetas = np.array(sample_prior_uniform(subkey, batch_size, prior_ranges))
        else:
            # Stop with fewer samples than requested once the bank is used up.
            if simulation_index >= num_precomputed:
                break
            end_index = min(simulation_index + batch_size, num_precomputed)
            thetas = all_thetas[simulation_index:end_index]

        for offset, theta in enumerate(thetas):
            summary = simulate_summary(theta, simulation_index + offset)
            distance = float(distance_fn(summary, observed_summary))
            if distance <= epsilon:
                accepted.append(theta)
                accepted_distances.append(distance)
                if len(accepted) >= num_samples:
                    break

        if simulations is not None:
            simulation_index += len(thetas)

        if num_loops % 10 == 0:
            proposals_done = (num_loops * batch_size) if simulations is None else simulation_index
            print(
                f"[ABC] Attempts (loops): {num_loops}, proposals processed: {proposals_done}, "
                f"accepted {len(accepted)}/{num_samples}"
            )

    # -------------------------------------------------------------
    # =================== ↑ Rejection loop ↑ ======================
    # -------------------------------------------------------------

    print(f"[ABC] Finished: accepted {len(accepted)} samples in {time() - start_time:.1f}s (loops: {num_loops})")

    accepted = np.array(accepted)
    accepted_distances = np.array(accepted_distances)
    np.savez_compressed(
        save_path,
        posterior=accepted,
        distances=accepted_distances,
        epsilon=epsilon,
    )
    return accepted, accepted_distances, epsilon
