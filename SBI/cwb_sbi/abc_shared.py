"""
Code shared with the ABC analysis (ABC/abc_inference): the code-unit system,
the prior ranges, the code-to-physical conversions and the posterior corner
plot. Both analyses use the same parameter space, so they share one
implementation rather than keeping copies in sync.
"""

# general
import sys

# local
from cwb_sbi.paths import ABC_DIRECTORY

if ABC_DIRECTORY not in sys.path:
    sys.path.append(ABC_DIRECTORY)

from abc_inference.plotting import plot_marginals  # noqa: E402
from abc_inference.priors import get_prior_ranges  # noqa: E402
from abc_inference.units import code_units, convert_theta_to_phys  # noqa: E402

__all__ = [
    "code_units",
    "convert_theta_to_phys",
    "get_prior_ranges",
    "plot_marginals",
]
