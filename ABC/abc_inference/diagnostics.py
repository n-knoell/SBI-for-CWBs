"""
Posterior diagnostics: 68% credible-interval widths compared to the prior, and
a printed report including a LaTeX table snippet.
"""

# numerics
import numpy as np

# local
from abc_inference.units import LAST_MASS_LOSS_RATE_INDEX, convert_to_phys


PARAMETER_LABELS = [
    "log M_dot_1",
    "log M_dot_2",
    "v_inf_1",
    "v_inf_2",
    "e",
    "cos i",
    "eta",
]

# Posterior / prior width ratios above this are reported as "essentially the
# prior", i.e. the parameter is not constrained by the data.
PRIOR_DOMINATED_RATIO = 0.85


def _to_display_units(value_in_code_units, parameter_index):
    """Convert to the table display units (log10 M_sun / yr for mass-loss rates)."""
    value = convert_to_phys(value_in_code_units, parameter_index)
    if parameter_index <= LAST_MASS_LOSS_RATE_INDEX:
        value = np.log10(value)
    return value


def posterior_68_ci_width(posterior, prior_ranges):
    """
    Compute the 68% credible-interval width per parameter, in the display units
    of the LaTeX table (Table 4):

        idx 0, 1 (mass-loss rates):  log10(M_sun / yr)
        idx 2, 3 (wind velocities):  km / s
        idx 4    (eccentricity):     dimensionless
        idx 5    (cos i):            dimensionless
        idx 6    (eta):              dimensionless

    Args:
        posterior: Accepted samples of shape [N, D] in code units.
        prior_ranges: List of (low, high) per parameter in code units.

    Returns:
        A list of dicts (one per parameter) with keys 'label', 'width_post',
        'width_prior' and 'ratio' = width_post / width_prior. Ratios close to 1
        indicate the posterior is essentially the prior.
    """
    rows = []
    for parameter_index in range(posterior.shape[1]):
        samples = _to_display_units(posterior[:, parameter_index], parameter_index)
        quantile_16, quantile_84 = np.percentile(samples, [16, 84])
        width_posterior = quantile_84 - quantile_16

        # For a uniform prior the 68% CI width is 0.68 times its range.
        prior_lower = _to_display_units(prior_ranges[parameter_index][0], parameter_index)
        prior_upper = _to_display_units(prior_ranges[parameter_index][1], parameter_index)
        width_prior = 0.68 * (prior_upper - prior_lower)

        rows.append({
            "label": PARAMETER_LABELS[parameter_index],
            "width_post": width_posterior,
            "width_prior": width_prior,
            "ratio": width_posterior / width_prior,
        })
    return rows


def report_widths(posterior, prior_ranges, name=""):
    """
    Print the 68% CI widths as a table and as a LaTeX table snippet.

    Args:
        posterior: Accepted samples of shape [N, D] in code units.
        prior_ranges: List of (low, high) per parameter in code units.
        name: Name of the observation, used in the headings.

    Returns:
        The rows computed by posterior_68_ci_width.
    """
    rows = posterior_68_ci_width(posterior, prior_ranges)

    print(f"\n=== ABC posterior 68% CI widths ({name}) ===")
    print(f"{'parameter':<14} {'width(post)':>12} {'width(prior)':>13} "
          f"{'post/prior':>11}  note")
    print("-" * 64)
    for row in rows:
        note = "≈ prior" if row["ratio"] > PRIOR_DOMINATED_RATIO else ""
        print(f"{row['label']:<14} {row['width_post']:>12.4f} "
              f"{row['width_prior']:>13.4f} {row['ratio']:>11.3f}  {note}")

    print(f"\n--- snippet ({name}) ---")
    for row in rows:
        if row["ratio"] > PRIOR_DOMINATED_RATIO:
            print(f"  {row['label']:<14} & --- & prior     & $\\gg 1$ \\\\")
        else:
            print(f"  {row['label']:<14} & --- & {row['width_post']:>8.2f} & --- \\\\")
    return rows
