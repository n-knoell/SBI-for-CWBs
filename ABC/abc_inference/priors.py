"""
Uniform prior ranges of the inferred parameters, in the code units the
simulator expects.
"""

# jax
import jax.numpy as jnp

# units and constants
from astropy import units as u

# local
from abc_inference.units import code_units


def get_prior_ranges():
    """
    Assemble the uniform prior box over the seven model parameters. The bounds
    are defined in physical units and converted to code units; the mass-loss
    rates are sampled in natural log.

    Returns:
        A list of (low, high) tuples, one per parameter in theta order:
        log Ṁ_1, log Ṁ_2, v_inf_1, v_inf_2, e, cos(i), η.
    """

    # -------------------------------------------------------------
    # ================ ↓ Bounds in physical units ↓ ===============
    # -------------------------------------------------------------

    mass_loss_rate_lower = 8e-9 * u.M_sun / u.yr
    mass_loss_rate_upper = 1e-5 * u.M_sun / u.yr
    wind_velocity_lower = 1200 * u.km / u.s
    wind_velocity_upper = 3200 * u.km / u.s

    eccentricity_lower = 0.0
    eccentricity_upper = 0.85
    cos_inclination_lower = 0.0
    cos_inclination_upper = 1.0
    turbulence_strength_lower = 0.0
    turbulence_strength_upper = 0.025

    # -------------------------------------------------------------
    # ================ ↑ Bounds in physical units ↑ ===============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================ ↓ Conversion to code units ↓ ===============
    # -------------------------------------------------------------

    code_mass_loss_rate_unit = code_units.code_mass / code_units.code_time
    mass_loss_rate_lower = mass_loss_rate_lower.to(code_mass_loss_rate_unit).value
    mass_loss_rate_upper = mass_loss_rate_upper.to(code_mass_loss_rate_unit).value
    wind_velocity_lower = float(wind_velocity_lower.to(code_units.code_velocity).value)
    wind_velocity_upper = float(wind_velocity_upper.to(code_units.code_velocity).value)

    log_mass_loss_rate_lower = float(jnp.log(mass_loss_rate_lower))
    log_mass_loss_rate_upper = float(jnp.log(mass_loss_rate_upper))

    # -------------------------------------------------------------
    # ================ ↑ Conversion to code units ↑ ===============
    # -------------------------------------------------------------

    return [
        (log_mass_loss_rate_lower, log_mass_loss_rate_upper),  # log Ṁ_1
        (log_mass_loss_rate_lower, log_mass_loss_rate_upper),  # log Ṁ_2
        (wind_velocity_lower, wind_velocity_upper),  # v_inf_1
        (wind_velocity_lower, wind_velocity_upper),  # v_inf_2
        (eccentricity_lower, eccentricity_upper),  # e
        (cos_inclination_lower, cos_inclination_upper),  # cos(i)
        (turbulence_strength_lower, turbulence_strength_upper),  # η
    ]
