"""
Robust evaluation of posterior log-probabilities for test-set metrics.
"""

# neural networks
import torch


def compute_log_prob_safe(posterior, theta, x, chunk_size=100):
    """
    Evaluate log p(theta_i | x_i) for a test set, falling back from the full
    batch to chunks and, for chunks that fail, to single samples.

    Args:
        posterior: The NPE posterior.
        theta: Parameters, shape (N, D).
        x: Observations, shape (N, ...).
        chunk_size: Chunk size of the first fallback.

    Returns:
        1D tensor of length N with the unnormalised log-probabilities.
    """
    try:
        return posterior.log_prob_batched(theta, x=x, norm_posterior=False).reshape(-1)
    except Exception:
        pass

    log_probs = []
    for start in range(0, theta.shape[0], chunk_size):
        theta_chunk = theta[start:start + chunk_size]
        x_chunk = x[start:start + chunk_size]
        try:
            log_prob_chunk = posterior.log_prob_batched(theta_chunk, x=x_chunk, norm_posterior=False)
            log_probs.append(log_prob_chunk.reshape(-1))
        except AssertionError:
            for index in range(theta_chunk.shape[0]):
                theta_single = theta_chunk[index:index + 1]
                x_single = x_chunk[index:index + 1]
                try:
                    log_prob_single = posterior.log_prob(theta_single, x=x_single, norm_posterior=False)
                except Exception:
                    log_prob_single = posterior.log_prob_batched(theta_single, x=x_single, norm_posterior=False)
                log_probs.append(log_prob_single.reshape(-1))
    return torch.cat(log_probs, dim=0)
