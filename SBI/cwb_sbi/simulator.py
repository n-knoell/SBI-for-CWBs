"""
Forward simulator: 3D hydrodynamics of a colliding-wind binary with turbulent
forcing (jf1uids / astronomix), post-processed into Hα photon-count maps.

The simulation configuration and the homogeneous initial state are built once
at import, so importing this module is expensive.
"""

# general
import os

# jax
import jax
import jax.numpy as jnp
from jax import random

# numerics
import numpy as np

# units and constants
from astropy import units as u
import astropy.constants as c

# jf1uids constants
from jf1uids._physics_modules._binary._binary_options import NGP
from jf1uids.option_classes.simulation_config import (
    FORWARDS,
    MINMOD,
    MUSCL,
    OPEN_BOUNDARY,
    SPLIT,
)

# jf1uids containers
from jf1uids import SimulationConfig
from jf1uids import SimulationParams
from jf1uids import WindParams
from jf1uids._physics_modules._binary._binary_options import BinaryParams
from jf1uids._physics_modules._cooling.cooling_options import CoolingConfig
from jf1uids._physics_modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)
from jf1uids.option_classes import WindConfig
from jf1uids.option_classes.simulation_config import (
    BinaryConfig,
    BoundarySettings,
    BoundarySettings1D,
)

# jf1uids functions
from jf1uids import get_helper_data
from jf1uids import get_registered_variables
from jf1uids import time_integration
from jf1uids._physics_modules._binary._binary import binary_starting_orbits_at_phase
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids.option_classes.simulation_config import finalize_config

# local
from cwb_sbi.abc_shared import code_units
from cwb_sbi.halpha import build_j_map, project_theta_pi2_fast, sb_to_photons_code
from cwb_sbi.paths import DATA_DIRECTORY


# -------------------------------------------------------------
# =============== ↓ Simulation settings ↓ =====================
# -------------------------------------------------------------

GAMMA = 5 / 3
C_CFL = 0.4
DT_MAX = 0.1

BOX_SIZE = 0.34
NUM_CELLS = 64
NUM_TIMESTEPS = 1600

# Number of snapshots of the 5 yr evolution, including the initial condition.
NUM_SNAPSHOTS = 11

# The binary semi-major axis in code units (0.34 / 3.8 ≈ 0.09).
SEMI_MAJOR_AXIS = BOX_SIZE / 3.8

END_TIME = (5 * u.yr).to(code_units.code_time).value

# Homogeneous ambient medium.
AMBIENT_DENSITY = 2 * c.m_p / u.cm**3
AMBIENT_PRESSURE = 3e4 * u.K / u.cm**3 * c.k_B

CELL_SIZE = BOX_SIZE / NUM_CELLS

# -------------------------------------------------------------
# =============== ↑ Simulation settings ↑ =====================
# -------------------------------------------------------------


def _build_config():
    """The simulation configuration (before finalization)."""
    open_boundaries = BoundarySettings1D(
        left_boundary=OPEN_BOUNDARY,
        right_boundary=OPEN_BOUNDARY,
    )
    return SimulationConfig(
        runtime_debugging=False,
        first_order_fallback=True,
        progress_bar=True,
        dimensionality=3,
        num_ghost_cells=2,
        box_size=BOX_SIZE,
        num_cells=NUM_CELLS,
        split=SPLIT,
        limiter=MINMOD,
        time_integrator=MUSCL,
        cooling_config=CoolingConfig(
            cooling=False,
        ),
        wind_config=WindConfig(
            stellar_wind=True,
            num_injection_cells=2,
            trace_wind_density=False,
            real_wind_params=False,
        ),
        binary_config=BinaryConfig(
            binary=True,
            deposit_particles=NGP,
            central_object_only=False,
        ),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True,
        ),
        fixed_timestep=False,
        differentiation_mode=FORWARDS,
        num_timesteps=NUM_TIMESTEPS,
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        boundary_settings=BoundarySettings(
            open_boundaries,
            open_boundaries,
            open_boundaries,
        ),
    )


def _build_initial_state(config, registered_variables):
    """Homogeneous gas at rest with the ambient density and pressure."""
    grid_shape = (config.num_cells, config.num_cells, config.num_cells)
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.ones(grid_shape) * AMBIENT_DENSITY.to(code_units.code_density).value,
        velocity_x=jnp.zeros(grid_shape),
        velocity_y=jnp.zeros(grid_shape),
        velocity_z=jnp.zeros(grid_shape),
        gas_pressure=jnp.ones(grid_shape) * AMBIENT_PRESSURE.to(code_units.code_pressure).value,
    )


config = _build_config()
helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)
initial_state = _build_initial_state(config, registered_variables)
config = finalize_config(config, initial_state.shape)


# -------------------------------------------------------------
# ========= ↓ Empirical mass - mass-loss-rate relation ↓ ======
# -------------------------------------------------------------

# Stellar masses and their mass-loss rates (mlr0_array.npy, in M_sun / yr),
# from the rotating stellar-evolution grids of Ekström et al. (2012).
STELLAR_MASSES = np.array([12, 15, 20, 25, 32, 40, 60, 85, 120])
MASS_LOSS_RATES = np.load(os.path.join(DATA_DIRECTORY, "mlr0_array.npy"))

mass_values = jnp.array((STELLAR_MASSES * u.M_sun).to(code_units.code_mass).value)
mlr_values = jnp.array((MASS_LOSS_RATES * u.M_sun / u.yr).to(code_units.code_mass / code_units.code_time).value)

# Relative scatter of the stellar mass around the empirical relation.
STELLAR_MASS_SCATTER = 0.05

# -------------------------------------------------------------
# ========= ↑ Empirical mass - mass-loss-rate relation ↑ ======
# -------------------------------------------------------------


def _run_hydro(theta, key):
    """
    Run the hydrodynamic simulation for one parameter vector.

    Args:
        theta: (log Ṁ_1, log Ṁ_2, v_inf_1, v_inf_2, e, cos(i), η) in code
            units, mass-loss rates in natural log.
        key: jax PRNG key for the stellar masses and the turbulent forcing.

    Returns:
        (simulation result with the snapshots, the key passed on to the
        simulation).
    """
    mlr1 = jnp.exp(theta[0])
    mlr2 = jnp.exp(theta[1])
    v_inf1 = theta[2]
    v_inf2 = theta[3]
    eccentricity = theta[4]
    inclination_deg = jnp.arccos(theta[5]) * (180.0 / jnp.pi)
    turbulence_strength = theta[6]

    # The stellar masses follow from the mass-loss rates with a 5% scatter.
    mass_source1_mean = jnp.interp(mlr1, mlr_values, mass_values)
    mass_source2_mean = jnp.interp(mlr2, mlr_values, mass_values)
    mass_source1 = mass_source1_mean + STELLAR_MASS_SCATTER * mass_source1_mean * random.normal(key)
    key = random.split(key)[0]
    mass_source2 = mass_source2_mean + STELLAR_MASS_SCATTER * mass_source2_mean * random.normal(key)

    masses = jnp.array([mass_source1, mass_source2])
    orbits = binary_starting_orbits_at_phase(
        SEMI_MAJOR_AXIS,
        eccentricity,
        inclination_deg,
        mass_source1,
        mass_source2,
        0.0,
        true_anom_deg=0.0,
    )

    # The turbulent energy injection rate is η times the mean wind luminosity.
    wind_luminosity = 0.5 * (mlr1 + mlr2) / 2 * ((v_inf1 + v_inf2) / 2)**2
    energy_injection_rate = turbulence_strength * wind_luminosity

    params = SimulationParams(
        C_cfl=C_CFL,
        dt_max=DT_MAX,
        gamma=GAMMA,
        t_end=END_TIME,
        wind_params=WindParams(
            wind_mass_loss_rates=jnp.array([mlr1, mlr2]),
            wind_final_velocities=jnp.array([v_inf1, v_inf2]),
        ),
        turbulent_forcing_params=TurbulentForcingParams(
            energy_injection_rate=energy_injection_rate,
        ),
        binary_params=BinaryParams(
            masses=masses,
            binary_state=orbits,
        ),
    )

    result = time_integration(
        initial_state,
        config,
        params,
        helper_data,
        registered_variables,
        key=key,
    )
    return result, key


def _just_hydro_result(theta, key):
    """The raw hydrodynamic result (snapshots of the full state) for theta."""
    result, _ = _run_hydro(theta, key)
    return result


def snapshot_to_photon_counts(state):
    """
    Convert one hydrodynamic snapshot into a clean Hα photon-count map
    projected along z. Counts are clipped to at least one photon; detector
    noise is added at training time (see cwb_sbi.noise.add_fresh_noise).
    """
    density = state[registered_variables.density_index]
    pressure = state[registered_variables.pressure_index]
    emissivity = build_j_map(density, pressure)
    surface_brightness = project_theta_pi2_fast(emissivity, CELL_SIZE)
    photon_counts = sb_to_photons_code(surface_brightness)
    return jnp.clip(photon_counts, a_min=1.0, a_max=None)


def sample_simulation(theta, key):
    """
    Simulate the clean Hα photon-count time series for one parameter vector.

    Args:
        theta: Parameters in code units (see _run_hydro).
        key: jax PRNG key.

    Returns:
        Photon counts of shape (NUM_SNAPSHOTS, H, W), including the T = 0
        snapshot.
    """
    result, _ = _run_hydro(theta, key)
    return jax.vmap(snapshot_to_photon_counts)(result.states)
