"""
Handcrafted, physics-motivated summary statistics of the Hα image time series
and their normalization over a batch of simulations.
"""

# jax
import jax.numpy as jnp

# numerics
import numpy as np


def physical_summary(x):
    """
    Physics-motivated handcrafted summary of a colliding-wind binary Hα time
    series. Per component, the physical interpretation is

        log(total flux)   [T]   mass-loss rates Ṁ (j_Hα ∝ n² ∝ Ṁ²)
        log(peak flux)    [T]   stagnation-point brightness / Ṁ ratio
        centroid cx, cy   [2T]  orbital motion, constrains e, i and phase
        spread σx, σy     [2T]  bow-shock opening angle, constrains v_inf
        flux std / mean   [1]   periastron variability, constrains e
        peak ratio        [1]   periastron brightening, constrains e

    Args:
        x: Photon counts of shape [T, H, W].

    Returns:
        A 1D array of length 6T + 2 (62 for T = 10).
    """
    x = jnp.asarray(x).astype(jnp.float64)
    num_frames, height, width = x.shape
    epsilon = 1e-12

    pixel_y, pixel_x = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.float64),
        jnp.arange(width, dtype=jnp.float64),
        indexing="ij",
    )

    # Per-frame flux statistics.
    total_flux = x.sum(axis=(1, 2))
    peak_flux = x.max(axis=(1, 2))
    normalization = total_flux + epsilon

    # The brightness-weighted centroid traces the orbital motion.
    centroid_x = (x * pixel_x).sum(axis=(1, 2)) / normalization
    centroid_y = (x * pixel_y).sum(axis=(1, 2)) / normalization

    # The brightness-weighted second moments trace the bow-shock extent.
    variance_x = ((pixel_x - centroid_x[:, None, None]) ** 2 * x).sum(axis=(1, 2)) / normalization
    variance_y = ((pixel_y - centroid_y[:, None, None]) ** 2 * x).sum(axis=(1, 2)) / normalization
    spread_x = jnp.sqrt(variance_x)
    spread_y = jnp.sqrt(variance_y)

    # The global temporal variability is an eccentricity signature.
    flux_std_over_mean = total_flux.std() / (total_flux.mean() + epsilon)
    peak_ratio = peak_flux.max() / (peak_flux.min() + epsilon)

    return jnp.concatenate([
        jnp.log10(total_flux + epsilon),
        jnp.log10(peak_flux + epsilon),
        centroid_x,
        centroid_y,
        spread_x,
        spread_y,
        jnp.array([flux_std_over_mean, peak_ratio]),
    ])


def compute_summary_norm(sim_array, summary_fn, max_sims=1000):
    """
    Precompute the per-component mean and standard deviation of the summary
    statistics over a batch of simulations, used to z-score the summaries.

    Args:
        sim_array: Simulated observations, indexable along the first axis.
        summary_fn: Function mapping one observation to its summary vector.
        max_sims: Maximum number of simulations used for the estimate.

    Returns:
        (mean, std) as jax arrays of the summary length.
    """
    num_used = min(max_sims, len(sim_array))
    summaries = np.stack(
        [np.asarray(summary_fn(sim_array[i])) for i in range(num_used)],
        axis=0,
    )
    mean = summaries.mean(axis=0)

    # A small offset avoids division by zero for constant components.
    std = summaries.std(axis=0) + 1e-8

    return jnp.asarray(mean), jnp.asarray(std)
