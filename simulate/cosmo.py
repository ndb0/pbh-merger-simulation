import numpy as np

# --- Physical and Cosmological Constants ---
# All constants are in SI units.

# Fundamental Constants
G = 6.67430e-11      # Unit: m^3 kg^-1 s^-2
c = 299792458.0      # Unit: m s^-1

# Astronomical Unit Conversions
M_sun_SI = 1.98847e30      # Unit: kg
parsec = 3.085677581491367e16  # Unit: m
Mpc = parsec * 1e6              # Unit: m
YEAR_S = 3.154e7                # Unit: s

# Planck 2018 Baseline Cosmological Parameters
H0_km_s_Mpc = 67.66  # Unit: km s^-1 Mpc^-1
H0 = H0_km_s_Mpc * 1000 / Mpc  # Unit: s^-1
Omega_m = 0.3111     # Matter density parameter (dimensionless)
Omega_r = 9.236e-5     # Radiation density parameter (dimensionless)
Omega_L = 1.0 - Omega_m - Omega_r  # Dark energy density (dimensionless)

H0_s = H0  # Hubble constant in s^-1

T_0_S = 13.8e9 * YEAR_S # Age of the Universe | Unit: s

# --- Derived Cosmological Values ---

# Critical density of the universe today
rho_crit_0 = (3.0 * H0_s**2) / (8.0 * np.pi * G) # Unit: kg m^-3

# Dark matter density today
# Assuming Omega_m is ~Omega_DM. A more precise calc would subtract Omega_baryon.
rho_DM_0 = Omega_m * rho_crit_0 # Unit: kg m^-3


# --- Epoch of Matter-Radiation Equality ---
z_eq = 3400.0        # Redshift at equality (dimensionless)
a_eq = 1.0 / (1.0 + z_eq) # Scale factor at equality (dimensionless)

# Time of Matter-Radiation Equality
# t_eq | Unit: s
# This formula is an approximation: t(a) = integral( da / (a * H(a)) )
# Let's use the explicit formulas
# H(a) = H0 * sqrt(Omega_r/a^4 + Omega_m/a^3 + ...)
# At early times, H(a) approx H0 * sqrt(Omega_r) / a^2
# a_dot = a * H(a) = H0 * sqrt(Omega_r) / a
# a * da = H0 * sqrt(Omega_r) * dt
# 0.5 * a^2 = H0 * sqrt(Omega_r) * t
# t = a^2 / (2 * H0 * sqrt(Omega_r))
t_eq = (a_eq**2) / (2.0 * H0_s * np.sqrt(Omega_r)) # Unit: s

def a_of_t(t):
    """Calculates the scale factor a for a given time t (in seconds)."""
    t = np.asarray(t) # [s]
    a = np.empty_like(t, dtype=float) # [dimensionless]
    rad_mask = t <= t_eq # [bool]
    mat_mask = ~rad_mask # [bool]
    # Radiation-dominated era: a ~ t^(1/2)
    a[rad_mask] = a_eq * np.sqrt(t[rad_mask] / t_eq)
    # Matter-dominated era: a ~ t^(2/3)
    # We must match the value and derivative at t_eq
    # a(t) = (a_eq / t_eq^(2/3)) * t^(2/3)
    # Let's use the simple textbook relation
    a[mat_mask] = a_eq * (t[mat_mask] / t_eq)**(2/3)
    return a # [dimensionless]

def t_of_a(a):
    """Calculates the time t (in seconds) for a given scale factor a."""
    a = np.asarray(a) # [dimensionless]
    t = np.empty_like(a, dtype=float) # [s]
    rad_mask = a <= a_eq # [bool]
    mat_mask = ~rad_mask # [bool]
    # Radiation-dominated era: t ~ a^2
    t[rad_mask] = t_eq * (a[rad_mask] / a_eq)**2
    # Matter-dominated era: t ~ a^(3/2)
    t[mat_mask] = t_eq * (a[mat_mask] / a_eq)**(1.5)
    return t # [s]

def z_of_a(a):
    """Calculates redshift z from scale factor a."""
    return 1.0 / a - 1.0 # [dimensionless]

def a_of_z(z):
    """Calculates scale factor a from redshift z."""
    return 1.0 / (1.0 + z) # [dimensionless]

def get_time_grid(variable="log_a", start=-20, end=0, steps=1000):
    """Generates time, scale factor, and redshift grids for the simulation."""
    if variable == "log_a":
        a_grid = np.exp(np.linspace(start, end, steps)) # [dimensionless]
        t_grid = t_of_a(a_grid) # [s]
    elif variable == "log_t":
        t_grid = np.exp(np.linspace(start, end, steps)) # [s]
        a_grid = a_of_t(t_grid) # [dimensionless]
    else:
        raise ValueError("Time grid variable must be 'log_a' or 'log_t'")
    z_grid = z_of_a(a_grid) # [dimensionless]
    return t_grid, a_grid, z_grid
    
def H_z(z):
    """Hubble parameter H(z) in s^-1."""
    return H0_s * np.sqrt(Omega_m * (1 + z)**3 + Omega_r * (1 + z)**4 + Omega_L) # [s^-1]


