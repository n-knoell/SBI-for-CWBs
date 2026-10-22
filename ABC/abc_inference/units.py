"""
Code-unit system of the colliding-wind binary simulations and the conversion of
the inferred parameters from code units to physical display units.

The parameter vector theta is ordered as

    0, 1: log mass-loss rates of the two stars (natural log, code units)
    2, 3: terminal wind velocities of the two stars (code units)
    4:    eccentricity e (dimensionless)
    5:    cos(i), cosine of the inclination (dimensionless)
    6:    turbulence strength η (dimensionless)
"""

# numerics
import numpy as np

# units and constants
from astropy import units as u
import astropy.constants as c

# jf1uids containers
from jf1uids import CodeUnits


# The binary separation in AU that sets the code length scale. The simulation
# box places the stars at a code separation of 0.1, so one code length unit
# corresponds to separation_in_au / 0.1 AU. Sensible range: 5 to 80 AU.
SEPARATION_IN_AU = 10

# Code mass unit in solar masses.
CODE_MASS_IN_SOLAR_MASSES = 0.01

# Indices of the parameter groups in theta, see the module docstring.
LAST_MASS_LOSS_RATE_INDEX = 1
LAST_WIND_VELOCITY_INDEX = 3


def build_code_units():
    """
    Build the code-unit system used by the simulations that produced the
    training / reference data. The velocity unit is chosen such that G = 1 in
    code units.

    Returns:
        The CodeUnits of the simulations.
    """
    code_length = SEPARATION_IN_AU / 0.1 * u.au
    code_mass = CODE_MASS_IN_SOLAR_MASSES * u.M_sun
    code_velocity = np.sqrt(c.G * code_mass / code_length).to(u.km / u.s)
    return CodeUnits(code_length, code_mass, code_velocity)


code_units = build_code_units()


def convert_to_phys(value_in_code_units, parameter_index):
    """
    Convert a parameter (or an array of samples of it) from code units to the
    physical units used for plotting.

    Args:
        value_in_code_units: Scalar or array of the parameter in code units.
        parameter_index: Index of the parameter in theta (see module docstring).

    Returns:
        The mass-loss rates in M_sun / yr (exponentiated from the log), the
        wind velocities in km / s, and the dimensionless parameters unchanged.
    """
    if parameter_index <= LAST_MASS_LOSS_RATE_INDEX:
        mass_loss_rate_unit_in_solar_masses_per_year = (
            1 * code_units.code_mass / code_units.code_time
        ).to(u.M_sun / u.yr).value
        return np.exp(value_in_code_units) * mass_loss_rate_unit_in_solar_masses_per_year
    elif parameter_index <= LAST_WIND_VELOCITY_INDEX:
        return (value_in_code_units * code_units.code_velocity).to(u.km / u.s).value
    else:
        return value_in_code_units


def convert_theta_to_phys(theta):
    """
    Convert a single parameter vector from code units to physical units.

    Args:
        theta: Parameter vector in code units (theta order, see module docstring).

    Returns:
        A list of the parameters in physical units.
    """
    return [
        convert_to_phys(value, index)
        for index, value in enumerate(theta)
    ]
