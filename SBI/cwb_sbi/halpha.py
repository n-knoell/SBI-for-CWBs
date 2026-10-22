"""
Hα post-processing of the hydrodynamic snapshots, entirely in code units:
the Hα volume emissivity from (ρ, P), its line-of-sight projection, the
conversion of surface brightness to detected photon counts, and the detector
noise model.
"""

# jax
import jax
import jax.numpy as jnp
from jax import jit, random, vmap

# units and constants
from astropy import units as u
import astropy.constants as const

# local
from cwb_sbi.abc_shared import code_units


# -------------------------------------------------------------
# ============== ↓ Physical constants in code units ↓ =========
# -------------------------------------------------------------

# The emissivity conversion is done with the code length and mass as plain
# quantities (100 AU and 0.01 M_sun), exactly as in the original pipeline.
_CODE_LENGTH = 10 / 0.1 * u.au
_CODE_MASS = 0.01 / 1 * u.M_sun

_code_energy = _CODE_MASS * _CODE_LENGTH**2 / code_units.code_time**2
_code_emissivity_unit = _code_energy / code_units.code_time / _CODE_LENGTH**3 / u.arcsec
_physical_emissivity_unit = u.erg / u.s / u.cm**3 / u.arcsec
_EMISSIVITY_TO_CODE = (1 * _physical_emissivity_unit).to(_code_emissivity_unit).value

PLANCK_CONSTANT = const.h.to(code_units.code_mass * code_units.code_length**2 / code_units.code_time).value
SPEED_OF_LIGHT = const.c.to(code_units.code_velocity).value
BOLTZMANN_CONSTANT = const.k_B.to(code_units.code_pressure * code_units.code_length**3 / u.K).value
PROTON_MASS = const.m_p.to(code_units.code_mass).value

MEAN_MOLECULAR_WEIGHT = 0.61
HYDROGEN_MASS_FRACTION = 0.71
HELIUM_MASS_FRACTION = 0.27

# T = P / (ρ / (μ m_p / k_B)) for an ideal gas.
_INVERSE_SPECIFIC_GAS_CONSTANT = (MEAN_MOLECULAR_WEIGHT * PROTON_MASS) / BOLTZMANN_CONSTANT

# -------------------------------------------------------------
# ============== ↑ Physical constants in code units ↑ =========
# -------------------------------------------------------------


# -------------------------------------------------------------
# =================== ↓ Hα emissivity ↓ =======================
# -------------------------------------------------------------

# Prefactor of j_Hα = 2.63e-33 n_e n_H / T^0.9 [erg cm^-3 s^-1 arcsec^-2]
# (Osterbrock 1989) for a fully photoionised H II region.
HALPHA_EMISSIVITY_PREFACTOR_PHYSICAL = 2.63e-33

_code_length_in_cm = (1.0 * code_units.code_length).to(u.cm).value
_code_volume_in_cm3 = _code_length_in_cm**3

_EMISSIVITY_PREFACTOR_CODE = _EMISSIVITY_TO_CODE * HALPHA_EMISSIVITY_PREFACTOR_PHYSICAL / (_code_volume_in_cm3**2)
_EMISSIVITY_PREFACTOR_PER_PROTON_MASS_SQUARED = _EMISSIVITY_PREFACTOR_CODE / (PROTON_MASS**2)

# Free electrons per hydrogen atom with fully ionised H and singly ionised He.
_ELECTRONS_PER_HYDROGEN = 1.0 + (HELIUM_MASS_FRACTION / (2.0 * HYDROGEN_MASS_FRACTION))


@jit
def build_j_map(rho, P):
    """
    Compute the Hα volume emissivity, staying entirely in code units.

    Args:
        rho: Density field (code units).
        P: Pressure field (code units).

    Returns:
        The emissivity map in code emissivity units.
    """
    temperature = P / (rho / _INVERSE_SPECIFIC_GAS_CONSTANT)
    hydrogen_density = rho * HYDROGEN_MASS_FRACTION
    return (
        _EMISSIVITY_PREFACTOR_PER_PROTON_MASS_SQUARED
        * (hydrogen_density**2)
        * _ELECTRONS_PER_HYDROGEN
        / (temperature**0.9)
    )

# -------------------------------------------------------------
# =================== ↑ Hα emissivity ↑ =======================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============ ↓ Line-of-sight projection ↓ ===================
# -------------------------------------------------------------


@jit
def grid_params_from_1d(x1, y1, z1):
    """Grid sizes, spacings, bounds and center from the 1D cell coordinates."""
    nx = x1.shape[0]; ny = y1.shape[0]; nz = z1.shape[0]
    x0, y0, z0 = x1[0], y1[0], z1[0]
    dx = x1[1] - x1[0] if nx > 1 else 1.0
    dy = y1[1] - y1[0] if ny > 1 else 1.0
    dz = z1[1] - z1[0] if nz > 1 else 1.0
    bounds_min = jnp.array([x1[0], y1[0], z1[0]])
    bounds_max = jnp.array([x1[-1], y1[-1], z1[-1]])
    center = (bounds_min + bounds_max) / 2.0
    return dict(nx=nx, ny=ny, nz=nz, x0=x0, y0=y0, z0=z0,
                dx=dx, dy=dy, dz=dz, bounds_min=bounds_min,
                bounds_max=bounds_max, center=center)


@jit
def trilinear_sample(emiss, params, pts):
    """
    Trilinearly interpolate a 3D field at arbitrary points; points outside the
    grid get zero.

    Args:
        emiss: Field of shape (nx, ny, nz).
        params: Grid parameters from grid_params_from_1d.
        pts: Points of shape (M, 3).

    Returns:
        The interpolated values, shape (M,).
    """
    x0, y0, z0 = params['x0'], params['y0'], params['z0']
    dx, dy, dz = params['dx'], params['dy'], params['dz']
    nx, ny, nz = params['nx'], params['ny'], params['nz']

    fx = (pts[:, 0] - x0) / dx
    fy = (pts[:, 1] - y0) / dy
    fz = (pts[:, 2] - z0) / dz

    inside = (fx >= 0) & (fx <= (nx - 1)) & (fy >= 0) & (fy <= (ny - 1)) & (fz >= 0) & (fz <= (nz - 1))

    ix = jnp.floor(jnp.clip(fx, 0, nx - 2)).astype(jnp.int32)
    iy = jnp.floor(jnp.clip(fy, 0, ny - 2)).astype(jnp.int32)
    iz = jnp.floor(jnp.clip(fz, 0, nz - 2)).astype(jnp.int32)

    wx = (fx - ix.astype(jnp.float32)).reshape(-1, 1)
    wy = (fy - iy.astype(jnp.float32)).reshape(-1, 1)
    wz = (fz - iz.astype(jnp.float32)).reshape(-1, 1)

    def as_column(values):
        return values.reshape(-1, 1)

    # The 8 corners of the enclosing cell.
    c000 = as_column(emiss[ix,     iy,     iz    ])
    c100 = as_column(emiss[ix + 1, iy,     iz    ])
    c010 = as_column(emiss[ix,     iy + 1, iz    ])
    c001 = as_column(emiss[ix,     iy,     iz + 1])
    c101 = as_column(emiss[ix + 1, iy,     iz + 1])
    c011 = as_column(emiss[ix,     iy + 1, iz + 1])
    c110 = as_column(emiss[ix + 1, iy + 1, iz    ])
    c111 = as_column(emiss[ix + 1, iy + 1, iz + 1])

    c00 = c000 * (1 - wx) + c100 * wx
    c01 = c001 * (1 - wx) + c101 * wx
    c10 = c010 * (1 - wx) + c110 * wx
    c11 = c011 * (1 - wx) + c111 * wx

    c0 = c00 * (1 - wy) + c10 * wy
    c1 = c01 * (1 - wy) + c11 * wy

    out = (c0 * (1 - wz) + c1 * wz).reshape(-1)
    return out * inside.astype(out.dtype)


def _box_corners(params):
    """The 8 corners of the grid bounding box, shape (8, 3)."""
    bounds_min, bounds_max = params['bounds_min'], params['bounds_max']
    return jnp.stack(
        jnp.meshgrid(
            jnp.array([bounds_min[0], bounds_max[0]]),
            jnp.array([bounds_min[1], bounds_max[1]]),
            jnp.array([bounds_min[2], bounds_max[2]]),
            indexing='ij',
        ),
        axis=-1,
    ).reshape(-1, 3)


def make_ray_integrator(emiss, params, n_steps):
    """
    Build a jitted function integrating the field along one ray through the
    box with the trapezoidal rule.

    Args:
        emiss: Field of shape (nx, ny, nz).
        params: Grid parameters from grid_params_from_1d.
        n_steps: Number of sampling points along the ray.

    Returns:
        integrate(origin, direction) -> line integral (zero if the ray misses
        the box).
    """
    corners = _box_corners(params)

    @jit
    def integrate(origin, nvec):
        # The span of the box along the ray follows from projecting its corners.
        tvals = jnp.dot(corners - origin, nvec)
        tmin = jnp.min(tvals); tmax = jnp.max(tvals)
        dt_total = tmax - tmin

        def integrate_path():
            ts = jnp.linspace(tmin, tmax, n_steps)
            pts = origin[None, :] + ts[:, None] * nvec[None, :]
            vals = trilinear_sample(emiss, params, pts)
            dt = dt_total / (n_steps - 1)
            return jnp.sum(0.5 * (vals[:-1] + vals[1:])) * dt

        return jax.lax.cond(dt_total > 0.0, integrate_path, lambda: 0.0)

    return integrate


def project_at_theta_zaxis(emiss, x1, y1, z1, theta, nu=None, nv=None, n_steps=None):
    """
    Ray-trace the field with the line of sight rotated around the z axis.

    Args:
        emiss: Field of shape (nx, ny, nz).
        x1, y1, z1: 1D cell coordinates.
        theta: Angle (rad) between the line of sight and +y (0 -> +y,
            pi/2 -> +x).
        nu, nv: Image size; defaults to (nx, nz), i.e. vertical axis = z.
        n_steps: Sampling points per ray; defaults to max(nx, ny, nz).

    Returns:
        (image of shape (nv, nu), u coordinates (nu,), v coordinates (nv,)).
    """
    params = grid_params_from_1d(x1, y1, z1)
    nx, ny, nz = params['nx'], params['ny'], params['nz']
    if nu is None: nu = nx
    if nv is None: nv = nz
    if n_steps is None: n_steps = max(nx, ny, nz)

    # Line of sight in the x-y plane; u is perpendicular to it in that plane
    # and v = n x u points along +z.
    nvec = jnp.array([jnp.sin(theta), jnp.cos(theta), 0.0])
    uvec = jnp.array([-jnp.cos(theta), jnp.sin(theta), 0.0])
    uvec = uvec / (jnp.linalg.norm(uvec) + 1e-16)
    vvec = jnp.cross(nvec, uvec)
    vvec = vvec / (jnp.linalg.norm(vvec) + 1e-16)

    # The image extents are measured relative to the box center.
    corners = _box_corners(params)
    center = params['center']
    uvals = jnp.dot(corners - center, uvec)
    vvals = jnp.dot(corners - center, vvec)
    u_min, u_max = jnp.min(uvals), jnp.max(uvals)
    v_min, v_max = jnp.min(vvals), jnp.max(vvals)

    u_coords = jnp.linspace(u_min, u_max, nu)
    v_coords = jnp.linspace(v_min, v_max, nv)

    U, V = jnp.meshgrid(u_coords, v_coords, indexing='xy')
    origins = center[None, :] + U.reshape(-1, 1) * uvec[None, :] + V.reshape(-1, 1) * vvec[None, :]

    integrator = make_ray_integrator(emiss, params, n_steps)
    vals = vmap(lambda origin: integrator(origin, nvec))(origins)
    img = vals.reshape(nv, nu)
    return img, u_coords, v_coords


@jit
def project_theta_pi2_fast(emiss, dz):
    """
    Fast projection along z on the solver grid (the theta = pi/2 case),
    assuming a uniform spacing dz.

    Returns:
        The projected map of shape (nx, ny).
    """
    return jnp.sum(emiss, axis=2) * dz

# -------------------------------------------------------------
# ============ ↑ Line-of-sight projection ↑ ===================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============= ↓ Photon counts and detector ↓ ================
# -------------------------------------------------------------

# Representative instrument: 2.4 m telescope, 600 s exposure, 11% throughput,
# 0.04 arcsec^2 pixel aperture, observing Hα at 656.28 nm.
APERTURE_AREA_ARCSEC2 = 0.04
TELESCOPE_DIAMETER = (2.4 * u.m).to(code_units.code_length).value
EXPOSURE_TIME = (600.0 * u.s).to(code_units.code_time).value
THROUGHPUT = 0.11
HALPHA_WAVELENGTH = (656.28 * u.nm).to(code_units.code_length).value

_sqrt_inverse_planck_constant = (1.0 / PLANCK_CONSTANT)**0.5
_telescope_area = jnp.pi * (TELESCOPE_DIAMETER ** 2) / 4.0
_photon_energy_over_planck_constant = SPEED_OF_LIGHT / HALPHA_WAVELENGTH

# The overall 1/h is split into a 1/sqrt(h) on each factor, which keeps the
# intermediate values at moderate magnitudes in float32.
_COLLECTED_ENERGY_RATE_FACTOR = APERTURE_AREA_ARCSEC2 * _telescope_area * _sqrt_inverse_planck_constant
_PHOTON_ENERGY_FACTOR = _photon_energy_over_planck_constant * PLANCK_CONSTANT**0.5


@jax.jit
def sb_to_photons_code(surface_sb_code):
    """
    Convert surface brightness to detected photon counts, entirely in code
    units.

    Args:
        surface_sb_code: Surface brightness in code energy / code time /
            code length^2 / arcsec^2.

    Returns:
        Detected photons (dimensionless), same shape as the input.
    """
    energy_rate = surface_sb_code * _COLLECTED_ENERGY_RATE_FACTOR
    photons_per_time = energy_rate / _PHOTON_ENERGY_FACTOR
    return photons_per_time * THROUGHPUT * EXPOSURE_TIME


def add_detector_noise_jax(image_photons,
                           sky_photons_per_pixel=0.0,
                           exposure_time_s=600.0,
                           qe=1.0,
                           dark_current_e_per_pix_per_s=0.001,
                           read_noise_e_rms=5.0,
                           flat_field_rms=0.01,
                           gain_e_per_adu=1.0,
                           key=None,
                           return_adu=False):
    """
    Apply the detector noise model: multiplicative flat field, Poisson shot
    noise of source, sky and dark current, and Gaussian read noise.

    Args:
        image_photons: Expected source photons per pixel for the full exposure.
        sky_photons_per_pixel: Expected sky photons per pixel.
        exposure_time_s: Exposure time in seconds (for the dark current).
        qe: Quantum efficiency (photons -> electrons); 1 if already in the
            throughput.
        dark_current_e_per_pix_per_s: Dark current in electrons / pixel / s.
        read_noise_e_rms: Read noise in electrons RMS.
        flat_field_rms: Fractional RMS of the flat field.
        gain_e_per_adu: Gain, used if return_adu.
        key: jax.random.PRNGKey (required).
        return_adu: Return ADU instead of electrons.

    Returns:
        (noisy image in electrons or ADU, new PRNG key for further draws).
    """
    new_key, flat_key, poisson_key, read_key = random.split(key, 4)

    # Flat-field variation with mean 1, clipped to avoid unphysical gains.
    if flat_field_rms > 0.0:
        flat = random.normal(flat_key, shape=image_photons.shape) * jnp.asarray(flat_field_rms) + 1.0
        flat = jnp.clip(flat, 0.5, 1.5)
    else:
        flat = 1.0

    expected_photons = (image_photons + sky_photons_per_pixel) * flat
    expected_photons = jnp.clip(expected_photons, a_min=0.0)

    expected_electrons = expected_photons * jnp.asarray(qe)
    dark_expected = jnp.asarray(dark_current_e_per_pix_per_s) * jnp.asarray(exposure_time_s)
    expected_electrons = expected_electrons + dark_expected
    expected_electrons = jnp.clip(expected_electrons, a_min=0.0)

    noisy_electrons = random.poisson(poisson_key, lam=expected_electrons)

    if read_noise_e_rms > 0.0:
        read_noise = random.normal(read_key, shape=noisy_electrons.shape) * jnp.asarray(read_noise_e_rms)
    else:
        read_noise = 0.0

    noisy_electrons = noisy_electrons + read_noise
    noisy_electrons = jnp.clip(noisy_electrons, a_min=0.0)

    if return_adu:
        return noisy_electrons / jnp.asarray(gain_e_per_adu), new_key
    return noisy_electrons, new_key

# -------------------------------------------------------------
# ============= ↑ Photon counts and detector ↑ ================
# -------------------------------------------------------------
