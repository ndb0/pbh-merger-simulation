import numpy as np

# --- Physical and Cosmological Constants ---
# All constants are in SI units.
G = 6.67430e-11      # Unit: m^3 kg^-1 s^-2
c = 299792458.0      # Unit: m s^-1
M_sun_SI = 1.98847e30      # Unit: kg
parsec = 3.085677581491367e16  # Unit: m
Mpc = parsec * 1e6              # Unit: m
YEAR_S = 3.154e7                # Unit: s
H0_km_s_Mpc = 67.66  # Unit: km s^-1 Mpc^-1
H0 = H0_km_s_Mpc * 1000 / Mpc  # Unit: s^-1
Omega_m = 0.3111     # Matter density parameter (dimensionless)
Omega_r = 9.236e-5     # Radiation density parameter (dimensionless)
Omega_L = 1.0 - Omega_m - Omega_r  # Dark energy density (dimensionless)
H0_s = H0  # Hubble constant in s^-1
T_0_S = 13.8e9 * YEAR_S # Age of the Universe | Unit: s

rho_crit_0 = (3.0 * H0_s**2) / (8.0 * np.pi * G) # Unit: kg m^-3
rho_DM_0 = Omega_m * rho_crit_0 # Unit: kg m^-3

z_eq = 3400.0        # Redshift at equality (dimensionless)
a_eq = 1.0 / (1.0 + z_eq) # Scale factor at equality (dimensionless)
t_eq = (a_eq**2) / (2.0 * H0_s * np.sqrt(Omega_r)) # Unit: s

def a_of_t(t):
    t = np.asarray(t)
    a = np.empty_like(t, dtype=float)
    rad_mask = t <= t_eq
    mat_mask = ~rad_mask
    a[rad_mask] = a_eq * np.sqrt(t[rad_mask] / t_eq)
    a[mat_mask] = a_eq * (t[mat_mask] / t_eq)**(2/3)
    return a

def t_of_a(a):
    a = np.asarray(a)
    t = np.empty_like(a, dtype=float)
    rad_mask = a <= a_eq
    mat_mask = ~rad_mask
    t[rad_mask] = t_eq * (a[rad_mask] / a_eq)**2
    t[mat_mask] = t_eq * (a[mat_mask] / a_eq)**(1.5)
    return t

def z_of_a(a):
    return 1.0 / a - 1.0

def a_of_z(z):
    return 1.0 / (1.0 + z)

def get_time_grid(variable="log_a", start=-20, end=0, steps=1000):
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
    return H0_s * np.sqrt(Omega_m * (1 + z)**3 + Omega_r * (1 + z)**4 + Omega_L)
