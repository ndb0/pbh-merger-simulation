import numpy as np

# --- Physical and Cosmological Constants ---
# All constants are in SI units unless otherwise specified.

# Fundamental Constants
G = 6.67430e-11      # Unit: m^3 kg^-1 s^-2 | Dimension: L^3 M^-1 T^-2
c = 299792458.0      # Unit: m s^-1 | Dimension: L T^-1

# Astronomical Unit Conversions
parsec = 3.085677581491367e16  # Unit: m
Mpc = parsec * 1e6             # Unit: m

# --- Astronomical Unit Conversions ---
M_sun_SI = 1.98847e30      # Mass of the Sun | Unit: kg
parsec = 3.085677581491367e16  # Parsec | Unit: m
Mpc = parsec * 1e6              # Megaparsec | Unit: m
YEAR_S = 3.154e7                # Seconds in a year | Unit: s
# Planck 2018 Baseline Cosmological Parameters
H0_km_s_Mpc = 67.66  # Hubble constant | Unit: km s^-1 Mpc^-1
H0 = H0_km_s_Mpc * 1000 / Mpc  # Hubble constant | Unit: s^-1 | Dimension: T^-1
Omega_m = 0.3111     # Matter density parameter | Dimensionless
Omega_r = 9.236e-5     # Radiation density parameter | Dimensionless
Omega_L = 1.0 - Omega_m - Omega_r  # Dark energy density | Dimensionless

H0_s = H0_km_s_Mpc * 1000 / Mpc  # Hubble constant | Unit: s^-1


T_0_S = 13.8e9 * YEAR_S # Age of the Universe | Unit: s

# Epoch of Matter-Radiation Equality
z_eq = 3400.0        # Redshift at equality | Dimensionless
a_eq = 1.0 / (1.0 + z_eq) # Scale factor at equality | Dimensionless

# Time of Matter-Radiation Equality
# t_eq | Unit: s | Dimension: T
t_eq = (2.0 * H0 * np.sqrt(Omega_r))**-1 * a_eq**2

def a_of_t(t):
    """Calculates the scale factor a for a given time t (in seconds)."""
    t = np.asarray(t)
    a = np.empty_like(t, dtype=float)
    rad_mask = t <= t_eq
    mat_mask = ~rad_mask
    # Radiation-dominated era: a ~ t^(1/2)
    a[rad_mask] = a_eq * np.sqrt(t[rad_mask] / t_eq)
    # Matter-dominated era: a ~ t^(2/3)
    a[mat_mask] = a_eq * (t[mat_mask] / t_eq)**(2/3)
    return a

def t_of_a(a):
    """Calculates the time t (in seconds) for a given scale factor a."""
    a = np.asarray(a)
    t = np.empty_like(a, dtype=float)
    rad_mask = a <= a_eq
    mat_mask = ~rad_mask
    # Radiation-dominated era: t ~ a^2
    t[rad_mask] = t_eq * (a[rad_mask] / a_eq)**2
    # Matter-dominated era: t ~ a^(3/2)
    t[mat_mask] = t_eq * (a[mat_mask] / a_eq)**(1.5)
    return t

def z_of_a(a):
    """Calculates redshift z from scale factor a."""
    return 1.0 / a - 1.0

def a_of_z(z):
    """Calculates scale factor a from redshift z."""
    return 1.0 / (1.0 + z)

def get_time_grid(variable="log_a", start=-20, end=0, steps=1000):
    """Generates time, scale factor, and redshift grids for the simulation."""
    if variable == "log_a":
        a_grid = np.exp(np.linspace(start, end, steps))
        t_grid = t_of_a(a_grid)
    elif variable == "log_t":
        t_grid = np.exp(np.linspace(start, end, steps))
        a_grid = a_of_t(t_grid)
    else:
        raise ValueError("Time grid variable must be 'log_a' or 'log_t'")
    z_grid = z_of_a(a_grid)
    return t_grid, a_grid, z_grid
def H_z(z):
    """Hubble parameter H(z) in s^-1."""
    return H0_s * np.sqrt(Omega_m * (1 + z)**3 + Omega_r * (1 + z)**4 + Omega_L)
