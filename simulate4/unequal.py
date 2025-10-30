# pbh_simulation_unequal_mass.py
# Simulates PBH binary evolution and merger rate for UNEQUAL MASSES using MCMC.
# WARNING: The formation probability function PDF_unequal_mass MUST be implemented
#          with the correct physics from e.g., Raidal et al. (2019).

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad, dblquad # dblquad might be needed for PDF normalization
from scipy.interpolate import interp1d
import scipy.special # May be needed for PDF implementation (e.g., hyperu)
import emcee
import warnings
import os
import time # For timing the simulation

# --- Matplotlib Parameters ---
mpl.rcParams.update({'font.size': 18,'font.family':'sans-serif'})
# Add other mpl params as needed...

# --- Constants ---
G_N = 4.302e-3 #(pc/solar mass) (km/s)^2
G_N_Mpc = 1e-6 * G_N #(Mpc/solar mass) (km/s)^2 # Conversion from pc to Mpc
h = 0.678
Omega_DM = 0.1186/(h**2)
H0 = 100.0*h #(km/s) Mpc^-1
H0_peryr = 67.8*(3.24e-20)*(60*60*24*365)
ageUniverse = 13.799e9 #y
Omega_L = 0.692
Omega_m = 0.308
Omega_r = 9.3e-5
rho_critical_Mpc = 3.0*H0**2/(8.0*np.pi*G_N_Mpc) #Solar masses per Mpc^3
rho_critical_Gpc = rho_critical_Mpc * (1e3)**3 # Solar masses per Gpc^3

z_eq = 3375.0
rho_eq = 1512.0 #Solar masses per pc^3
sigma_eq = np.sqrt(0.005) # Sqrt from pbh_merger_rate.py, original used 0.005
lambda_max = 3.0
alpha = 0.1

# --- Utility Functions ---
def Hubble(z):
    return H0_peryr*np.sqrt(Omega_L + Omega_m*(1+z)**3 + Omega_r*(1+z)**4)

def t_univ(z):
    integrand = lambda x: 1.0/((1+x)*Hubble(x))
    # Integrate from z to a large value (effectively infinity)
    t_int, err = quad(integrand, z, np.inf, epsabs=1e-9, epsrel=1e-9)
    # Check for integration errors
    if not np.isfinite(t_int):
        print(f"Warning: t_univ integration failed for z={z}. Returning 0.")
        return 0.0
    # Convert from seconds^-1 units (via H0_peryr) back to years? Check units.
    # H0_peryr is in yr^-1. Integral is dimensionless. Result should be in years.
    return t_int

def Omega_PBH(f):
    return f*Omega_DM

def n_PBH_pair(f, M1, M2):
    # Number density of pairs - careful definition needed for unequal mass
    # Simplistic approach: Use total density, needs refinement based on theory
    rho_pbh_tot = rho_critical_Gpc * Omega_PBH(f)
    # This calculation assumes random pairing, may need modification based on formation model
    # Need number densities n1, n2 if mass function is not monochromatic
    # Placeholder: assume equal number density for M1, M2 type for simplicity
    n_total_approx = rho_pbh_tot / ((M1 + M2) / 2.0) # Very rough estimate
    # Probability of finding M1 near M2 might scale as n1*n2*dV ?
    # Let's use the provided code's implicit scaling in rate calculation instead.
    # Return total PBH density for now.
    return rho_pbh_tot


# === START: Physics Functions (Adapted for Unequal Mass) ===

# --- Binary Dynamics (from modified Sampling.py) ---
def t_coal(a, e, M1, M2):
    """Calculates coalescence time for a binary with masses M1, M2."""
    if a <= 0 or e < 0 or e >= 1 or M1 <= 0 or M2 <= 0: return 0.0
    M = M1 + M2
    if M <= 0: return 0.0
    mu = (M1 * M2) / M
    eta = mu / M
    if eta <= 0: return 0.0

    Q_mass_term = G_N**3 * M1 * M2 * M
    if Q_mass_term == 0: return 0.0
    Q_const_factor = (3.0/170.0)
    Q = Q_const_factor / Q_mass_term

    j_squared = 1.0 - e**2
    if j_squared <= 0: return 0.0
    j_pow_7 = j_squared**(3.5)

    tc_geom = Q * a**4 * j_pow_7
    tc_seconds = tc_geom * 3.086e13 * (3e5)**5
    if tc_seconds <= 0: return 0.0
    return tc_seconds / (60*60*24*365) # years

def j_coal(a, t_years, M1, M2):
    """Calculates dimensionless angular momentum j = sqrt(1-e^2) for merger at t_years."""
    if a <= 0 or t_years <= 0 or M1 <= 0 or M2 <= 0: return 0.0
    M = M1 + M2
    if M <= 0: return 0.0
    mu = (M1 * M2) / M
    eta = mu / M
    if eta <= 0: return 0.0

    Q_mass_term = G_N**3 * M1 * M2 * M
    if Q_mass_term == 0: return 0.0
    Q_const_factor = (3.0/170.0)
    Q = Q_const_factor / Q_mass_term

    tc_seconds = t_years * (60*60*24*365)
    tc_geom = tc_seconds / ((3e5)**5 * 3.086e13)

    if Q == 0 or a == 0: return 0.0
    j_pow_7 = tc_geom / (Q * a**4)

    if j_pow_7 <= 0: return 0.0
    if j_pow_7 >= 1.0: return 1.0 # Cap at circular

    # Calculate e^2 directly to avoid issues with 7th root of numbers near 1
    e_squared_calc = 1.0 - j_pow_7**(2.0/7.0)
    if e_squared_calc < 0 or e_squared_calc > 1:
        # print(f"Warning: e^2 calculation out of bounds ({e_squared_calc:.2e}) in j_coal. a={a:.2e}, t={t_years:.2e}, M1={M1}, M2={M2}, j^7={j_pow_7:.2e}")
        # Decide how to handle: return 0 (invalid), 1 (merges instantly?), or adjust
        return 1.0 if j_pow_7 > 0.5 else 0.0 # Heuristic: if j^7 is large, likely near circular?

    j = np.sqrt(1.0 - e_squared_calc)
    return min(j, 1.0)


# --- Halo Properties (Depend on single mass M_PBH - Keep signature) ---
# Copied from modified Remapping.py
def r_trunc(z, M_PBH):
    """Truncation radius."""
    if M_PBH <= 0: return 0.0
    r0 = 6.3e-3 # 1300 AU in pc

    # Ensure non-negative redshift using NumPy's maximum
    # z_phys = max(0.0, z) # <-- Original problematic line
    z_phys = np.maximum(0.0, z) # <-- Corrected line

    # Ensure denominator is not zero or negative (though 1+z_phys should be >= 1)
    denominator = 1. + z_phys
    # Handle array case for denominator check
    if isinstance(denominator, np.ndarray):
        if np.any(denominator <= 0):
            print("Warning: Non-positive denominator encountered in r_trunc.")
            # Set result to 0 where denominator is invalid
            result = np.zeros_like(denominator)
            valid_mask = denominator > 0
            result[valid_mask] = r0 * (M_PBH)**(1.0/3.0) * (1. + z_eq) / denominator[valid_mask]
            return result
        else:
            # All denominators are valid
            return r0 * (M_PBH)**(1.0/3.0) * (1. + z_eq) / denominator
    else: # Scalar case
        if denominator <= 0:
             print("Warning: Non-positive denominator encountered in r_trunc.")
             return 0.0
        else:
            return r0 * (M_PBH)**(1.0/3.0) * (1. + z_eq) / denominator

def r_eq(M_PBH):
    return r_trunc(z_eq, M_PBH)

# Log-prior - MODIFIED FOR M1, M2
def lnprior(theta, M1, M2, a1, a2, tmin_sampling=1.e8, tmax_sampling=1.e11):
    """Log prior probability for parameters theta=(log10(a), log10(j))."""
    la, lj = theta
    # Check if parameters result in finite physical values
    if not (np.isfinite(la) and np.isfinite(lj)):
         return -np.inf

    a = 10**la
    j = 10**lj

    if not (j > 0 and j <= 1.0): # Ensure j is strictly positive and <= 1
        return -np.inf

    if not (a >= a1 and a <= a2): # Use >= and <= for range limits
        return -np.inf

    # Calculate eccentricity, ensure it's valid
    e_squared = 1. - j**2
    if e_squared < 0 or e_squared >= 1: # e must be < 1 for bound orbits (excluding parabolic)
         return -np.inf
    e = np.sqrt(e_squared)

    # Use modified t_coal
    t = t_coal(a, e, M1=M1, M2=M2)

    # Check if t calculation was valid and within sampling range
    if t <= 0 or not np.isfinite(t): return -np.inf

    if (t < tmin_sampling or t > tmax_sampling):
        return -np.inf

    return 0 # Log prior is 0 if within bounds

# Log-probability - MODIFIED FOR M1, M2
def lnprob(theta, f, M1, M2, PDF_unequal, a1, a2):
    """Log posterior probability, combining prior and likelihood (PDF)."""
    # Use modified lnprior
    lp = lnprior(theta, M1, M2, a1, a2) # Assuming default tmin/tmax are ok
    if not np.isfinite(lp):
        return -np.inf

    la, lj = theta
    # Check for non-finite inputs early
    if not (np.isfinite(la) and np.isfinite(lj)):
         return -np.inf

    a = 10**la
    j = 10**lj

    # !!! CRUCIAL: PDF_unequal MUST be the unequal mass probability distribution !!!
    # This function needs to be implemented based on Raidal et al. 2019 or similar.
    # It should take (la, lj, f, M1, M2) or (a, j, f, M1, M2) as arguments.
    try:
        # Pass log values directly if PDF expects them
        pdf_val = PDF_unequal(la, lj, f, M1, M2)
        # Or pass a, j if PDF expects physical values:
        # pdf_val = PDF_unequal(a, j, f, M1, M2)
    except Exception as e:
        # Catch potential errors in the user-provided PDF
        # print(f"Warning: PDF function failed for la={la}, lj={lj}. Error: {e}")
        return -np.inf

    # Ensure PDF value is valid for log
    if pdf_val <= 0 or not np.isfinite(pdf_val):
        return -np.inf

    log_pdf_val = np.log(pdf_val)
    if not np.isfinite(log_pdf_val): # Check log result too
        return -np.inf

    return lp + log_pdf_val
# --- MCMC Sampling Function (from modified Sampling.py) ---
def GetSamples_MCMC(N_samps, PDF_unequal, a1, a2, f, M1, M2, nwalkers=100, burn_in_steps=1000, thin_by=5):
    """
    Runs MCMC sampler to get samples from the unequal mass PDF.

    Args:
        N_samps (int): Number of desired samples *after* thinning.
        PDF_unequal (callable): The unequal mass probability distribution function.
                                Should accept (la, lj, f, M1, M2).
        a1, a2 (float): Min and max semi-major axis for prior.
        f (float): PBH fraction.
        M1, M2 (float): Component masses.
        nwalkers (int): Number of MCMC walkers.
        burn_in_steps (int): Number of steps to discard as burn-in.
        thin_by (int): Factor to thin the chain by.

    Returns:
        numpy.ndarray: Array of samples (shape: [N_samps, 2]). Returns None on failure.
    """
    ndim = 2
    # Calculate total steps needed, ensuring enough samples after thinning per walker
    steps_per_walker_needed = (N_samps // nwalkers + 1) * thin_by
    total_steps = steps_per_walker_needed + burn_in_steps # Add burn-in steps to this

    if total_steps <= burn_in_steps:
        print(f"Warning: Total steps ({total_steps}) calculation potentially problematic. Adjusting.")
        total_steps = burn_in_steps + max(100, N_samps // nwalkers * thin_by) # Ensure some steps after burn-in


    # --- Initial Guess ---
    a0 = np.sqrt(a1 * a2) # Geometric mean for a
    t0_guess = ageUniverse # Use age of Universe for j guess
    j0 = j_coal(a0, t0_guess, M1=M1, M2=M2)

    # Handle case where initial guess is invalid
    if j0 <= 1e-9 or j0 >= 1.0: # Use small threshold for > 0 check
        # print(f"Warning: Initial j0 calculation invalid or boundary value ({j0:.2e}) for a0={a0:.2e}, M1={M1}, M2={M2}. Trying different initial t.")
        # Try a different time, maybe closer to t_min or t_max of prior?
        j0_alt = j_coal(a0, 1e9, M1=M1, M2=M2) # Try 1 Gyr
        if j0_alt <= 1e-9 or j0_alt >=1.0:
             j0 = 1e-3 # Fallback to a small value
             # print(f"Warning: Alternative j0 also invalid. Using default j0={j0:.1e}.")
        else:
             j0 = j0_alt
             # print(f"Using alternative j0 = {j0:.2e}")
    j0 = max(1e-9, min(j0, 0.99999)) # Ensure it's strictly within (0, 1)


    # --- Initial positions for walkers ---
    p0 = []
    attempts = 0
    max_attempts = nwalkers * 200 # Increased attempts limit

    print(f"   Finding {nwalkers} valid starting positions...")
    while len(p0) < nwalkers and attempts < max_attempts:
        attempts += 1
        # Start close to a likely region
        a_start = a0 * (1 + 0.2 * (np.random.rand() - 0.5)) # Slightly wider spread for a
        # Ensure a_start is within prior bounds a1, a2
        a_start = max(a1, min(a_start, a2))

        j_start = j0 * (1 + 0.8 * (np.random.rand() - 0.5)) # Wider spread for j
        j_start = max(1e-9, min(j_start, 0.99999)) # Keep j within (0, 1) bounds

        # Check if this starting point has finite probability
        la_start, lj_start = np.log10(a_start), np.log10(j_start)
        initial_log_prob = lnprob([la_start, lj_start], f, M1, M2, PDF_unequal, a1, a2)

        if np.isfinite(initial_log_prob):
            p0.append([la_start, lj_start])

    if len(p0) < nwalkers:
        print(f"ERROR: Could not find {nwalkers} valid starting positions after {max_attempts} attempts. Check PDF or priors.")
        return None # Indicate failure

    p0 = np.array(p0)
    print(f"   Found {len(p0)} valid start positions.")

    # --- Initialize and run sampler ---
    print(f"   Starting MCMC run ({nwalkers} walkers, {total_steps} steps)...")
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[f, M1, M2, PDF_unequal, a1, a2])

        # Run MCMC with progress bar
        sampler.run_mcmc(p0, total_steps, progress=True)

        # --- Extract samples ---
        # Discard burn-in, flatten, and thin the chain
        samples = sampler.get_chain(discard=burn_in_steps, flat=True, thin=thin_by)

        if len(samples) < N_samps / 2: # Check if we got a reasonable number of samples
             print(f"Warning: MCMC yielded only {len(samples)} samples after thinning (requested {N_samps}). Chain might not have converged well.")
        elif len(samples) == 0:
             print("ERROR: MCMC yielded zero samples after thinning.")
             return None


        return samples

    except Exception as e:
        print(f"ERROR during emcee run: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None # Indicate failure

def M_halo(z, M_PBH):
    """DM halo mass."""
    if M_PBH <= 0: return 0.0
    req = r_eq(M_PBH)
    if req <= 0: return 0.0

    # Ensure z is non-negative using NumPy's maximum function
    # z_phys = max(0.0, z) # <-- Original problematic line
    z_phys = np.maximum(0.0, z) # <-- Corrected line

    rt = r_trunc(z_phys, M_PBH)
    # rt can now be an array if z was an array
    # Handle cases where elements of rt might be non-positive
    if isinstance(rt, np.ndarray):
         rt[rt <= 0] = 1e-99 # Use a tiny positive number to avoid ratio issues
    elif rt <= 0:
        return 0.0

    ratio = rt / req # req is a scalar
    # Handle potential negative ratios if rt had issues (now prevented)
    if isinstance(ratio, np.ndarray):
        ratio[ratio < 0] = 0.0
    elif ratio < 0:
        ratio = 0.0

    return M_PBH * (ratio)**1.5

def rho(r, r_tr, M_PBH, gamma=3.0/2.0):
    if r_tr <= 0 or r < 0: return 0.0
    x = r / r_tr
    req = r_eq(M_PBH)
    if req <= 0: return 0.0
    A_denom = (4 * np.pi * (r_tr**gamma) * (req**(3 - gamma)))
    if A_denom == 0: return 0.0
    A = (3 - gamma) * M_PBH / A_denom
    if (x <= 1):
        if x == 0 and gamma > 0: return np.inf
        return A * x**(-gamma)
    else: return 0

def Menc(r, r_tr, M_PBH, gamma=3.0/2.0):
    if M_PBH <= 0: return 0.0
    if r < 0: r = 0.0
    if r_tr <= 0: return M_PBH
    x = r / r_tr
    req = r_eq(M_PBH)
    if req <= 0: return M_PBH
    r_over_req = max(0.0, r / req)
    rtr_over_req = max(0.0, r_tr / req)
    exponent = 3.0 - gamma
    if (x <= 1):
        power_term = 0.0 if r_over_req == 0 and exponent > 0 else (1.0 if r_over_req == 0 and exponent == 0 else (np.inf if r_over_req == 0 and exponent < 0 else r_over_req**exponent))
    else:
        power_term = 0.0 if rtr_over_req == 0 and exponent > 0 else (1.0 if rtr_over_req == 0 and exponent == 0 else (np.inf if rtr_over_req == 0 and exponent < 0 else rtr_over_req**exponent))
    if not np.isfinite(power_term): power_term = 0.0
    return M_PBH * (1. + power_term)

def calcBindingEnergy(r_tr, M_PBH):
    if r_tr <= 0: return 0.0
    def integrand(r):
        density = rho(r, r_tr, M_PBH)
        enclosed_mass = Menc(r, r_tr, M_PBH)
        if not (np.isfinite(density) and np.isfinite(enclosed_mass)): return 0.0
        return enclosed_mass * density * r
    lower_bound = min(1e-9 * r_tr, r_tr*0.99) # Avoid exact boundaries if integrator struggles
    upper_bound = r_tr
    if upper_bound <= lower_bound: return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, error = quad(integrand, lower_bound, upper_bound, epsrel=1e-3, limit=100)
        if not np.isfinite(result): return 0.0
        return -G_N * 4 * np.pi * result
    except Exception: return 0.0


# --- Formation Physics Primitives (!!! PLACEHOLDERS - REQUIRE UNEQUAL MASS IMPLEMENTATION !!!) ---
# You MUST replace these with functions based on Raidal et al. 2019 or similar
def xbar(f, M1, M2):
    """Mean comoving separation. !!! PLACEHOLDER !!!"""
    # print("Warning: Using placeholder 'xbar' based on total mass.")
    M_eff = M1 + M2
    if rho_eq <= 0 or f <= 0 or M_eff <= 0: return 0.0
    term = (3.0 * M_eff / (4 * np.pi * rho_eq * (0.85 * f)))
    return (term)**(1.0 / 3.0) if term >= 0 else 0.0

def bigX(x, f, M1, M2):
     """Dimensionless separation cubed. !!! PLACEHOLDER !!!"""
     # print("Warning: Using placeholder 'bigX'.")
     xb = xbar(f, M1, M2)
     if xb == 0: return np.inf
     x_phys = max(0.0, x)
     return (x_phys / xb)**3.0

def x_of_a(a, f, M1, M2):
     """Comoving separation for semi-major axis a. !!! PLACEHOLDER !!!"""
     # print("Warning: Using placeholder 'x_of_a'.")
     xb = xbar(f, M1, M2)
     if alpha == 0 or f <= 0 or xb == 0 or a < 0: return 0.0
     term = (a * (0.85 * f) * xb**3.) / alpha
     return (term)**(1./4.) if term >= 0 else 0.0

def z_decoupling(a, f, M_calc1, M_calc2):
     """Decoupling redshift. !!! PLACEHOLDER !!!"""
     # print("Warning: Using placeholder 'z_decoupling'.")
     M_eff1, M_eff2 = M_calc1, M_calc2
     if a <= 0 or f <= 0 or M_eff1 <= 0 or M_eff2 <= 0: return -1.0
     x_val = x_of_a(a, f, M_eff1, M_eff2)
     X_val = bigX(x_val, f, M_eff1, M_eff2)
     denom_term = (0.85 * f)
     if denom_term == 0: return -1.0
     denom = (1./3. * X_val / denom_term)
     if denom <= 0: return -1.0
     z_dec = (1. + z_eq) / denom - 1.
     return max(-1.0, z_dec)

def a_max(f, M1, M2, withHalo=False):
     """Max semi-major axis. !!! PLACEHOLDER !!!"""
     # print("Warning: Using placeholder 'a_max'.")
     M_eff1, M_eff2 = M1, M2
     if withHalo:
         M_eff1 += M_halo(z_eq, M1)
         M_eff2 += M_halo(z_eq, M2)
     xb = xbar(f, M_eff1, M_eff2)
     if alpha == 0 or f <= 0 or xb == 0 : return 0.0
     term_ratio = lambda_max
     return alpha * xb * (f * 0.85)**(1.0/3.0) * (term_ratio**(4.0/3.0))

def a_max_with_Halo(f, M1, M2):
     """Max semi-major axis (with halo). !!! PLACEHOLDER !!!"""
     # print("Warning: Using placeholder 'a_max_with_Halo'.")
     return a_max(f, M1, M2, withHalo=True) # Use the placeholder a_max with halo flag


# --- Interpolator Management (from modified Remapping.py) ---
interpolator_cache = {}

def get_rtr_interpolator(M_PBH):
    """Generates the r_tr(a) interpolation function for a specific mass M_PBH."""
    # print(f"   Generating r_tr(a) interpolator for M_PBH = {M_PBH}...") # Reduce print frequency
    f_calc = 0.01
    try:
        am = a_max_with_Halo(f_calc, M_PBH, M_PBH) # Placeholder call
        if not np.isfinite(am) or am <= 0: raise ValueError("a_max calculation failed.")
    except Exception: am = 0.1 # Default
    a_list = np.logspace(-8, np.log10(am * 1.1), 51) # Reduced points

    z_decoupling_current = np.array([z_decoupling(a, f_calc, M_PBH, M_PBH) for a in a_list])
    z_decoupling_current[z_decoupling_current < 0] = 0.0
    M_halo_current = M_halo(z_decoupling_current, M_PBH)

    for _ in range(3):
        M_total_iter = M_PBH + M_halo_current
        M_total_iter[M_total_iter <= M_PBH] = M_PBH
        z_decoupling_next = np.array([z_decoupling(a, f_calc, mt, mt) for a, mt in zip(a_list, M_total_iter)])
        z_decoupling_next[z_decoupling_next < 0] = 0.0
        M_halo_next = M_halo(z_decoupling_next, M_PBH)
        z_decoupling_current = z_decoupling_next
        M_halo_current = M_halo_next

    final_z_decoupling = z_decoupling_current
    final_z_decoupling[final_z_decoupling < 0] = 0.0
    r_list = r_trunc(final_z_decoupling, M_PBH)

    valid_indices = np.isfinite(a_list) & np.isfinite(r_list) & (a_list > 0) & (r_list > 0)
    if not np.any(valid_indices): return None
    a_list_valid, r_list_valid = a_list[valid_indices], r_list[valid_indices]
    sort_indices = np.argsort(a_list_valid)
    a_list_sorted, r_list_sorted = a_list_valid[sort_indices], r_list_valid[sort_indices]
    unique_a, unique_indices = np.unique(a_list_sorted, return_index=True)
    if len(unique_a) < 2: return None
    try:
        return interp1d(unique_a, r_list_sorted[unique_indices], kind='linear', fill_value="extrapolate", bounds_error=False)
    except ValueError: return None

def get_Ubind_interpolator(M_PBH):
    """Generates the U_bind(r_tr) interpolation function for a specific mass M_PBH."""
    # print(f"   Generating U_bind(r_tr) interpolator for M_PBH = {M_PBH}...") # Reduce print frequency
    req = r_eq(M_PBH)
    if req <= 0: return None
    r_tr_min = max(1e-9, req * 1e-6) # Ensure positive min
    r_tr_max = req * 1.0
    if r_tr_min >= r_tr_max: return None
    rtr_vals = np.logspace(np.log10(r_tr_min), np.log10(r_tr_max), 50) # Reduced points
    Ubind_vals = np.array([calcBindingEnergy(r1, M_PBH) for r1 in rtr_vals])

    valid_indices = np.isfinite(rtr_vals) & np.isfinite(Ubind_vals) & (rtr_vals > 0)
    if not np.any(valid_indices): return None
    rtr_vals_valid, Ubind_vals_valid = rtr_vals[valid_indices], Ubind_vals[valid_indices]
    sort_indices = np.argsort(rtr_vals_valid)
    rtr_vals_sorted, Ubind_vals_sorted = rtr_vals_valid[sort_indices], Ubind_vals_valid[sort_indices]
    unique_rtr, unique_indices = np.unique(rtr_vals_sorted, return_index=True)
    if len(unique_rtr) < 2: return None
    try:
        return interp1d(unique_rtr, Ubind_vals_sorted[unique_indices], kind='linear', fill_value="extrapolate", bounds_error=False)
    except ValueError: return None

def setup_interpolators(M1, M2):
    """Generates and caches interpolators for M1 and M2 if not already done."""
    global interpolator_cache
    success = True
    masses_to_setup = {M1, M2}
    for M_val in masses_to_setup:
        if M_val not in interpolator_cache or interpolator_cache[M_val]['rtr'] is None or interpolator_cache[M_val]['Ubind'] is None:
            # print(f"Setting up interpolators for M_PBH = {M_val}...") # Reduce print frequency
            rtr_interp = get_rtr_interpolator(M_val)
            Ubind_interp = get_Ubind_interpolator(M_val)
            if rtr_interp is None or Ubind_interp is None:
                 print(f"Error: Failed to generate interpolators for M={M_val}")
                 interpolator_cache[M_val] = {'rtr': None, 'Ubind': None}; success = False
            else: interpolator_cache[M_val] = {'rtr': rtr_interp, 'Ubind': Ubind_interp}
    return success

# --- Remapping Functions (from modified Remapping.py) ---
def calc_af(ai, M1, M2):
    """Calculates final semi-major axis after halo ejection."""
    global G_N, interpolator_cache
    if ai <= 0 or M1 <= 0 or M2 <= 0: return ai if ai > 0 else np.inf
    if not setup_interpolators(M1, M2): return ai
    cache_M1 = interpolator_cache.get(M1); cache_M2 = interpolator_cache.get(M2)
    if cache_M1 is None or cache_M2 is None or cache_M1['rtr'] is None or cache_M1['Ubind'] is None or cache_M2['rtr'] is None or cache_M2['Ubind'] is None: return ai
    rtr_interp_M1, Ubind_interp_M1 = cache_M1['rtr'], cache_M1['Ubind']
    rtr_interp_M2, Ubind_interp_M2 = cache_M2['rtr'], cache_M2['Ubind']

    try: r_tr_1 = float(rtr_interp_M1(ai)); r_tr_1 = max(1e-9, r_tr_1)
    except ValueError: r_tr_1 = max(rtr_interp_M1.x[0], 1e-9) if len(rtr_interp_M1.x)>0 else 1e-9
    M_tot_1 = Menc(r_tr_1, r_tr_1, M1)
    try: U_bind_1 = float(Ubind_interp_M1(r_tr_1)); U_bind_1 = 0.0 if not np.isfinite(U_bind_1) else U_bind_1
    except ValueError: U_bind_1 = 0.0

    try: r_tr_2 = float(rtr_interp_M2(ai)); r_tr_2 = max(1e-9, r_tr_2)
    except ValueError: r_tr_2 = max(rtr_interp_M2.x[0], 1e-9) if len(rtr_interp_M2.x)>0 else 1e-9
    M_tot_2 = Menc(r_tr_2, r_tr_2, M2)
    try: U_bind_2 = float(Ubind_interp_M2(r_tr_2)); U_bind_2 = 0.0 if not np.isfinite(U_bind_2) else U_bind_2
    except ValueError: U_bind_2 = 0.0

    U_orb_before = -G_N * M_tot_1 * M_tot_2 / (2.0 * ai)
    U_orb_final = U_orb_before + U_bind_1 + U_bind_2
    if U_orb_final >= -1e-15: return np.inf
    af_denom = (2.0 * U_orb_final)
    if af_denom == 0: return np.inf
    af = -G_N * M1 * M2 / af_denom
    return af if af > 0 else np.inf

def calc_jf(ji, ai, M1, M2):
    """Calculates final dimensionless angular momentum jf after halo ejection."""
    global interpolator_cache
    if not (0 < ji <= 1.0) or ai <= 0 or M1 <= 0 or M2 <= 0: return ji if (0 < ji <= 1.0) else 0.0
    if not setup_interpolators(M1, M2): return ji
    cache_M1 = interpolator_cache.get(M1); cache_M2 = interpolator_cache.get(M2)
    if cache_M1 is None or cache_M2 is None or cache_M1['rtr'] is None or cache_M2['rtr'] is None: return ji
    rtr_interp_M1, rtr_interp_M2 = cache_M1['rtr'], cache_M2['rtr']

    try: r_tr_1 = float(rtr_interp_M1(ai)); r_tr_1 = max(1e-9, r_tr_1)
    except ValueError: r_tr_1 = max(rtr_interp_M1.x[0], 1e-9) if len(rtr_interp_M1.x)>0 else 1e-9
    M_tot_1 = Menc(r_tr_1, r_tr_1, M1)
    try: r_tr_2 = float(rtr_interp_M2(ai)); r_tr_2 = max(1e-9, r_tr_2)
    except ValueError: r_tr_2 = max(rtr_interp_M2.x[0], 1e-9) if len(rtr_interp_M2.x)>0 else 1e-9
    M_tot_2 = Menc(r_tr_2, r_tr_2, M2)

    af = calc_af(ai, M1, M2)
    if not np.isfinite(af) or af <= 0: return 0.0

    M_i = M_tot_1 + M_tot_2; M_f = M1 + M2
    if M_i <= 0 or M_f <= 0: return ji
    mu_i = (M_tot_1 * M_tot_2) / M_i if M_i > 0 else 0.0
    mu_f = (M1 * M2) / M_f if M_f > 0 else 0.0
    if mu_f <= 0: return ji

    sqrt_term_num = M_i * ai; sqrt_term_den = M_f * af
    if sqrt_term_den == 0: return 0.0
    sqrt_term = sqrt_term_num / sqrt_term_den
    if sqrt_term < 0: return 0.0

    jf = ji * (mu_i / mu_f) * np.sqrt(sqrt_term)
    return max(0.0, min(jf, 1.0))

# === START: Formation Probability (!!! CRITICAL PLACEHOLDER SECTION !!!) ===
# You MUST replace these functions with the correct physics derived from
# Raidal et al. 2019 (arXiv:1812.01930) or similar unequal mass papers.

def PDF_unequal_mass(la, lj, f, M1, M2):
    """
    Placeholder for the log-probability density P(log10(a), log10(j))
    for unequal masses M1, M2 at fraction f.

    !!! REPLACE THIS WITH THE ACTUAL PHYSICS IMPLEMENTATION !!!
    This might involve Eq. 2.31 or 2.32 from Raidal et al. 2019 ,
    calculating j0 (Eq. 2.21 ), N(y) (Eq. 3.1 or 3.5 ), sigma_M,
    and potentially integrating over the mass function psi(m).

    This placeholder returns a constant value, which is WRONG.
    """
    # print("Warning: Using placeholder PDF_unequal_mass function!")
    # Simple example: Return 1 if j is small, 0 otherwise (VERY WRONG)
    # j = 10**lj
    # return 1.0 if j < 0.1 else 1e-9 # Needs proper physics

    # Placeholder based on scaling in original P_a_j * a * j * (np.log(10)**2)
    # This maintains some structure but physics is still from equal mass.
    a = 10**la
    j = 10**lj
    if j <=0 or j > 1 or a <= 0: return 1e-99 # Avoid log(0)

    # Use original equal-mass logic as placeholder structure
    M_eff = (M1 + M2) / 2.0 # Very rough guess
    try:
        xval = x_of_a(a, f, M1, M2, withHalo = True) # Using placeholder x_of_a
        X = bigX(xval, f, M1, M2) # Using placeholder bigX
        xb = xbar(f, M1, M2) # Using placeholder xbar

        # Placeholder j_X equivalent
        # Need average mass <m> and <m^2> for correct j0 and sigma_j (Eq 2.21, 2.26 [cite: 248-250, 283-284])
        # Using a simplistic j0 scaling as placeholder
        avM = M_eff # Placeholder for average mass
        j0_placeholder = 0.4 * f / (X / (xval/xb)**3) if X > 0 else 1e-3 # Very rough scaling derived from combining defs
        jX_placeholder = j0_placeholder * 0.5 * (1 + sigma_eq**2 / (0.85 * f)**2)**0.5 # Based on original code structure

        # Placeholder P_j equivalent
        y = j / jX_placeholder if jX_placeholder > 0 else np.inf
        Pj_placeholder = (y**2 / (1 + y**2)**(1.5)) / j if j > 0 else 0

        # Placeholder measure equivalent
        measure_placeholder = (3.0/4.0)*(a**-0.25)*(0.85*f/(alpha*xb))**0.75 if alpha > 0 and xb > 0 else 0
        # Simplistic halo mass factor placeholder
        z_dec_approx = z_decoupling(a, f, M1, M2) # Placeholder
        Mh1 = M_halo(z_dec_approx, M1); Mh2 = M_halo(z_dec_approx, M2)
        halo_factor = ((M1+Mh1 + M2+Mh2)/(M1+M2))**(3./4.) if (M1+M2)>0 else 1.0
        measure_placeholder *= halo_factor

        # Combine placeholders
        prob_val = Pj_placeholder * np.exp(-X) * measure_placeholder

        # Convert P(a, j) to P(la, lj)
        pdf_la_lj = prob_val * a * j * (np.log(10)**2)

        # Return a small positive value if calculation fails or is zero
        return max(pdf_la_lj, 1e-99) if np.isfinite(pdf_la_lj) else 1e-99

    except Exception as e:
        # print(f"Warning: Placeholder PDF calculation failed. a={a:.2e}, j={j:.2e}. Error: {e}")
        return 1e-99 # Return tiny value on error

# --- MCMC Sampling Setup (from modified Sampling.py) ---
# Assuming lnprior, lnprob, GetSamples_MCMC use the unequal mass versions defined above

# --- Main Simulation Loop ---

# Define parameters
Nsamples_per_run = 2**12 # Reduced for faster testing (was 2**16)
amin_glob = 5.e-5 # Global minimum a
tmin_sampling = 1.e8
tmax_sampling = 1.e11

# Define mass pairs and f_pbh values to simulate
# Example: Using the peak masses from user request
mass_pairs = [
    (1.0, 1.0),       # M1-M1
    (1e-4, 1e-4),     # M2-M2
    (1.0, 1e-4)      # M1-M2 (and M2-M1, result should be symmetric)
]
f_pbh_values = np.logspace(-4., -1., 5) # Reduced number of f values for speed

# Store results
results = {}

print("Starting PBH Binary Evolution Simulation for Unequal Masses...")

start_time_total = time.time()

for m1, m2 in mass_pairs:
    print(f"\n*** Simulating Mass Pair: M1 = {m1:.1e} M_sun, M2 = {m2:.1e} M_sun ***")
    pair_key = f"M1_{m1:.1e}_M2_{m2:.1e}"
    results[pair_key] = {}

    # Setup interpolators for this mass pair ONCE
    print("Setting up interpolators...")
    if not setup_interpolators(m1, m2):
        print(f"ERROR: Cannot proceed with pair ({m1}, {m2}) due to interpolator failure.")
        continue
    print("Interpolators ready.")

    for f_val in f_pbh_values:
        print(f"\n--- Running for f_PBH = {f_val:.3e} ---")
        f_key = f"f_{f_val:.3e}"
        results[pair_key][f_key] = {}

        start_time_f = time.time()

        # Determine a_max for sampling range (use placeholder)
        # Needs careful implementation based on formation theory for M1, M2
        try:
            amax_sim = a_max_with_Halo(f_val, m1, m2) # Using placeholder
            if not np.isfinite(amax_sim) or amax_sim <= amin_glob:
                print(f"Warning: amax calculation failed or invalid ({amax_sim:.2e}). Using default 0.1")
                amax_sim = 0.1
        except Exception as e:
            print(f"Error calculating amax: {e}. Using default 0.1")
            amax_sim = 0.1

        print(f"Sampling PDF for a in [{amin_glob:.1e}, {amax_sim:.3e}] pc...")
        try:
            # --- Run MCMC Sampling ---
            # Pass the placeholder PDF - REPLACE THIS in lnprob or here
            samples_MCMC = GetSamples_MCMC(Nsamples_per_run, PDF_unequal_mass,
                                          amin_glob, amax_sim, f_val, m1, m2,
                                          nwalkers=50, burn_in_steps=500, thin_by=10) # Faster params

            if samples_MCMC is None or len(samples_MCMC) == 0:
                 print("ERROR: MCMC sampling failed to return valid samples.")
                 results[pair_key][f_key]['error'] = "MCMC failed"
                 continue

            la_vals_all = samples_MCMC[:,0]
            lj_vals_all = samples_MCMC[:,1]
            num_actual_samples = len(la_vals_all)
            print(f"... MCMC done! Got {num_actual_samples} samples.")

            # --- Calculate Initial and Remapped Coalescence Times ---
            t_vals_initial = np.zeros(num_actual_samples)
            t_vals_remapped = np.zeros(num_actual_samples)
            valid_sample_count = 0

            print("Calculating initial and remapped coalescence times...")
            for ind in range(num_actual_samples):
                a_i = 10.**(la_vals_all[ind])
                j_i = 10.**(lj_vals_all[ind])

                if not (j_i > 0 and j_i <= 1.0): continue # Skip invalid samples
                e_i_squared = 1. - j_i**2
                if e_i_squared < 0 or e_i_squared >= 1.0: continue
                e_i = np.sqrt(e_i_squared)

                # Initial time
                t_initial = t_coal(a_i, e_i, m1, m2)
                t_vals_initial[ind] = t_initial

                # Remapped parameters
                a_f = calc_af(a_i, m1, m2)
                if not np.isfinite(a_f) or a_f <= 0:
                    t_vals_remapped[ind] = np.inf # Unbound or error
                    continue

                j_f = calc_jf(j_i, a_i, m1, m2)
                if not (j_f >= 0 and j_f <= 1.0): # Allow jf=0
                     t_vals_remapped[ind] = np.inf # Invalid jf
                     continue

                e_f_squared = 1. - j_f**2
                # Allow e_f = 1 (j_f = 0) -> infinite time
                if e_f_squared < 0 or e_f_squared > 1.0:
                    t_vals_remapped[ind] = np.inf # Invalid e_f
                    continue
                # If j_f is very close to 0, e_f is very close to 1
                if j_f < 1e-9: # Effectively radial orbit
                     t_remapped = np.inf
                else:
                    e_f = np.sqrt(e_f_squared)
                    t_remapped = t_coal(a_f, e_f, m1, m2)

                t_vals_remapped[ind] = t_remapped
                if np.isfinite(t_remapped) and t_remapped > 0:
                    valid_sample_count += 1

            print(f"... Calculation done! {valid_sample_count} valid remapped finite times.")

            # --- Store Results ---
            results[pair_key][f_key]['samples_a_initial'] = 10**la_vals_all
            results[pair_key][f_key]['samples_j_initial'] = 10**lj_vals_all
            results[pair_key][f_key]['t_initial'] = t_vals_initial
            results[pair_key][f_key]['t_remapped'] = t_vals_remapped
            results[pair_key][f_key]['num_samples'] = num_actual_samples

            # --- Calculate Merger Rate (Simplified Example) ---
            # The rate calculation requires normalizing the PDF and considering the fraction
            # of binaries that merge today. This requires dblquad for normalization,
            # and is complex with MCMC samples.
            # Simplified approach: Fraction of *sampled* binaries merging around t0.
            t_today = ageUniverse
            t_window = 0.1 * t_today # +/- 10% window

            merging_now_initial = np.sum((t_vals_initial >= t_today - t_window) & (t_vals_initial <= t_today + t_window))
            merging_now_remapped = np.sum((t_vals_remapped >= t_today - t_window) & (t_vals_remapped <= t_today + t_window))

            frac_initial = merging_now_initial / num_actual_samples if num_actual_samples > 0 else 0
            frac_remapped = merging_now_remapped / num_actual_samples if num_actual_samples > 0 else 0

            # Estimate rate: Need total formation rate * fraction merging now.
            # Total formation rate depends on integral of PDF. VERY complex.
            # Placeholder Rate Estimation: Rate ~ n_pairs * (fraction/t_window) ? Needs theory.
            # Using scaling similar to analytical code for comparison: R ~ f^(53/37) * M_factor * S * frac_factor
            # This is highly approximate.
            M_avg = (m1+m2)/2.0 # Simplistic average mass
            eta = (m1*m2)/(m1+m2)**2 if (m1+m2)>0 else 0

            # Use analytical code's S_E, S_L placeholders if needed for scaling comparison
            # These were defined in pbh_merger_rate.py - copy them here if needed
            # Or use S_E=1, S_L=1 for unsuppressed comparison
            S_L_approx = 1.0 # Placeholder
            S_E_approx = 1.0 # Placeholder

            rate_scale_factor = 1.6e6 * (f_val**(53.0/37.0)) * (eta**(-34.0/37.0)) * ((m1+m2)**(-32.0/37.0)) * S_L_approx * S_E_approx

            # Rate ~ scale * (fraction merging / characteristic time scale related to sampling)
            # This is not rigorous. A proper rate calculation needs the normalized PDF integral.
            # Let's just store the fractions for now.
            rate_initial_approx = rate_scale_factor * frac_initial # Very rough scaling
            rate_remapped_approx = rate_scale_factor * frac_remapped # Very rough scaling

            results[pair_key][f_key]['fraction_merging_initial'] = frac_initial
            results[pair_key][f_key]['fraction_merging_remapped'] = frac_remapped
            # results[pair_key][f_key]['rate_initial_approx'] = rate_initial_approx
            # results[pair_key][f_key]['rate_remapped_approx'] = rate_remapped_approx

            print(f"Fraction merging near t0 (initial): {frac_initial:.3g}")
            print(f"Fraction merging near t0 (remapped): {frac_remapped:.3g}")
            # print(f"Approx Rate Scaling (initial): {rate_initial_approx:.3g}")
            # print(f"Approx Rate Scaling (remapped): {rate_remapped_approx:.3g}")


        except Exception as e:
            print(f"ERROR during simulation for f={f_val:.3e}: {e}")
            results[pair_key][f_key]['error'] = str(e)

        end_time_f = time.time()
        print(f"--- Time for f={f_val:.3e}: {end_time_f - start_time_f:.1f} seconds ---")


end_time_total = time.time()
print(f"\n*** Total Simulation Time: {(end_time_total - start_time_total)/60.0:.2f} minutes ***")

# --- Post-processing / Plotting Example ---
# Plot histogram of coalescence times for the last run pair/f
if 'samples_a_initial' in results[pair_key][f_key]: # Check if last run succeeded
    t_plot_initial = results[pair_key][f_key]['t_initial']
    t_plot_remapped = results[pair_key][f_key]['t_remapped']

    # Filter out NaNs/Infs for histogramming
    t_plot_initial_finite = t_plot_initial[np.isfinite(t_plot_initial) & (t_plot_initial > 0)]
    t_plot_remapped_finite = t_plot_remapped[np.isfinite(t_plot_remapped) & (t_plot_remapped > 0)]

    if len(t_plot_initial_finite) > 0 and len(t_plot_remapped_finite) > 0:
        plt.figure(figsize=(8, 6))
        bins = np.logspace(np.log10(max(1e7, np.min(t_plot_initial_finite))),
                           np.log10(min(1e13, np.max(t_plot_initial_finite))), 50)

        plt.hist(t_plot_initial_finite, bins=bins, alpha=0.6, label='Initial $\\tau_{coal}$', density=True)
        plt.hist(t_plot_remapped_finite, bins=bins, alpha=0.6, label='Remapped $\\tau_{coal}$', density=True)

        plt.axvline(ageUniverse, color='k', linestyle='--', label='Age of Universe')
        plt.xscale('log')
        plt.yscale('log') # Or linear: plt.yscale('linear')
        plt.xlabel('Coalescence Time $\\tau_{coal}$ [yr]')
        plt.ylabel('Normalized Distribution $P(\\tau)$')
        plt.title(f'Coalescence Time Distribution ({pair_key}, {f_key})')
        plt.legend()
        plt.grid(True, which='both', ls=':')
        plt.tight_layout()
        plot_filename = f"coalescence_times_{pair_key}_{f_key}.png"
        plt.savefig(plot_filename)
        print(f"Saved coalescence time histogram to {plot_filename}")
        # plt.show() # If running interactively
        plt.close()
    else:
        print("Not enough valid finite coalescence times to plot histogram for the last run.")

# --- Save results (optional) ---
# import pickle
# with open('simulation_results_unequal_mass.pkl', 'wb') as f:
#     pickle.dump(results, f)
# print("Simulation results saved to simulation_results_unequal_mass.pkl")

print("\nScript Finished.")

# === END: Physics Functions ===

# --- Need definition for a_max, a_max_with_halo if used in GetSamples_MCMC ---
# These are placeholders from Remapping.py - MUST BE REPLACED
# def a_max(f, M1, M2, withHalo=False): ...
# def a_max_with_Halo(f, M1, M2): ...
