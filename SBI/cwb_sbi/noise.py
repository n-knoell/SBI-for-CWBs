"""
Fresh detector-noise realisations for clean photon-count simulations, so that
training can resample the noise on the same underlying simulations.
"""

# jax
import jax
import jax.numpy as jnp
from jax import random as jrandom

# numerics
import numpy as np

# neural networks
import torch

# local
from cwb_sbi.halpha import add_detector_noise_jax


def add_fresh_noise(x_clean_torch, seed, chunk=500):
    """
    Draw a fresh detector-noise realisation for a batch of clean photon-count
    snapshots. The noisy counts are clipped to at least one photon.

    Args:
        x_clean_torch: Clean photon counts, torch tensor of shape (N, T, H, W).
        seed: Seed of the jax PRNG.
        chunk: Number of samples pushed through jax at once.

    Returns:
        Noisy photon counts of shape (N, T, H, W), with the dtype and device of
        the input.
    """
    original_device = x_clean_torch.device
    original_dtype = x_clean_torch.dtype

    x_clean = x_clean_torch.detach().cpu().numpy()
    num_samples, num_frames, _, _ = x_clean.shape

    def noise_one_frame(image, key):
        noisy_image, _ = add_detector_noise_jax(image, key=key)
        return jnp.clip(noisy_image, a_min=1.0, a_max=None)

    # vmap over samples and frames: (N, T, H, W) with keys (N, T, 2).
    noise_batch = jax.vmap(jax.vmap(noise_one_frame))

    key = jrandom.PRNGKey(seed)
    noisy_chunks = []
    for start in range(0, num_samples, chunk):
        end = min(start + chunk, num_samples)
        clean_chunk = jnp.asarray(x_clean[start:end])
        chunk_size = end - start
        key, subkey = jrandom.split(key)
        frame_keys = jrandom.split(subkey, chunk_size * num_frames).reshape(chunk_size, num_frames, 2)
        noisy_chunks.append(np.asarray(noise_batch(clean_chunk, frame_keys)))

    x_noisy = np.concatenate(noisy_chunks, axis=0)
    return torch.from_numpy(x_noisy).to(device=original_device, dtype=original_dtype)
