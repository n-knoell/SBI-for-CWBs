"""
The uniform NPE prior over the seven model parameters.
"""

# neural networks
import torch

# simulation-based inference
from sbi.utils import BoxUniform

# local
from cwb_sbi.abc_shared import get_prior_ranges


def make_prior(device=None):
    """
    Build the BoxUniform prior over (log Ṁ_1, log Ṁ_2, v_inf_1, v_inf_2, e,
    cos(i), η) in code units, with the mass-loss rates in natural log.

    Args:
        device: Torch device of the prior; None uses the sbi default.

    Returns:
        The BoxUniform prior.
    """
    prior_ranges = get_prior_ranges()
    return BoxUniform(
        low=torch.tensor([lower for lower, _ in prior_ranges]),
        high=torch.tensor([upper for _, upper in prior_ranges]),
        device=device,
    )
