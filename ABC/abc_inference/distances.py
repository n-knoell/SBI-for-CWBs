"""
Distances between a simulated and the observed summary vector.
"""

# jax
import jax.numpy as jnp


def standardized_l2(summ_sim, summ_obs, mean, std):
    """
    Root-mean-square distance in standardized (z-scored) summary space.

    Args:
        summ_sim: Summary vector of the simulation.
        summ_obs: Summary vector of the observation.
        mean: Per-component mean used for standardization.
        std: Per-component standard deviation used for standardization.

    Returns:
        The scalar distance.
    """
    standardized_simulation = (summ_sim - mean) / std
    standardized_observation = (summ_obs - mean) / std
    return jnp.sqrt(jnp.mean((standardized_simulation - standardized_observation) ** 2))


def weighted_standardized_l2(summ_sim, summ_obs, mean, std, weights):
    """
    Like standardized_l2, but with per-component weights on the squared
    differences.

    Args:
        summ_sim: Summary vector of the simulation.
        summ_obs: Summary vector of the observation.
        mean: Per-component mean used for standardization.
        std: Per-component standard deviation used for standardization.
        weights: Per-component weights.

    Returns:
        The scalar distance.
    """
    standardized_simulation = (summ_sim - mean) / std
    standardized_observation = (summ_obs - mean) / std
    return jnp.sqrt(jnp.mean(weights * (standardized_simulation - standardized_observation) ** 2))
