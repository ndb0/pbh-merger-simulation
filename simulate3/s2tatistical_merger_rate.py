import numpy as np
import os
import time
import argparse
import json
from tqdm import tqdm
from config import load_config
import cosmo
import emcee
from scipy.interpolate import interp1d
from scipy.integrate import quad, dblquad
from math import log10, sqrt, log, exp

# --- 1. Constants and Physics Functions (from Remapping.py & cosmo.py) ---
G_N = 4.302e-3 #(pc/solar mass) (km/s)^2
alpha = 0.1
z_eq = 3375.0
rho_eq = 1512.0 #Solar masses per pc^3
sigma_eq = 0.005 #Variance of DM density perturbations at equality
lambda_max = 3.0 #Max decoupling redshift parameter

# Global interpolators (will be initialized)
rtr_interp = None
Ubind_interp = None
current_MPBH = -10.0

# --- Cosmological Functions ---
H0_peryr = cosmo.H0_km_s_Mpc * (3.24e-20) * (60*60*24*365)
Omega_L = cosmo.Omega_L
Omega_m = cosmo.Omega_m
Omega_r = cosmo.Omega_r

def Hubble(z):
    return H0_peryr*np.sqrt(Omega_L + Omega_m*(1+z)**3 + Omega_r*(1+z)**4)

def t_univ(z):
    integ = lambda x: 1.0/((1+x)*Hubble(x))
    return quad(integ, z, np.inf)[0]

# --- Analytical PDF Functions (from Remapping.py) ---
def r_trunc(z, M_PBH):
    r0 = 6.3e-3 #1300 AU in pc
    return r0*(M_PBH)**(1.0/3.0)*(1.+z_eq)/(1.+z)

def r_eq(M_PBH):
    return r_trunc(z_eq, M_PBH)

def M_halo(z, M_PBH):
    # CRASH FIX: Vectorized check for z > 0
    z = np.clip(z, 1e-9, None)
    return M_PBH*(r_trunc(z, M_PBH)/r_eq(M_PBH))**1.5

def xbar(f, M_PBH):
    # M_PBH is the total mass of the binary M = m1 + m2
    return (3.0*M_PBH/(4*np.pi*rho_eq*(0.85*f)))**(1.0/3.0)

def bigX(x, f, M_PBH):
    return (x/(xbar(f,M_PBH)))**3.0

def z_decoupling(a, f, mass):
    # Calculate decoupling redshift
    x = x_of_a(a, f, mass)
    # CRASH FIX: Vectorized check
    x = np.clip(x, 1e-9, None) # x must be > 0

    X_val = bigX(x, f, mass)
    X_val = np.clip(X_val, 1e-9, None) # X_val must be > 0

    z_dec = (1. + z_eq)/(1./3 * X_val / (0.85*f)) - 1.
    return np.clip(z_dec, 0, None) # z cannot be negative

def x_of_a(a, f, M_PBH):
    xb = xbar(f, M_PBH)
    val = (a * (0.85*f) * xb**3.)/alpha
    # CRASH FIX: Vectorized check for val >= 0
    val = np.clip(val, 0, None)
    return val**(1./4.)

def a_max(f, M_PBH, withHalo = False):
    Mtot = 1.0*M_PBH
    if (withHalo):
        Mtot += M_halo(z_eq, M_PBH) # M_halo at z_eq
    return alpha*xbar(f, Mtot)*(f*0.85)**(1.0/3.0)*((lambda_max)**(4.0/3.0))

def GetRtrInterp(M_PBH):
    # M_PBH is the total mass of the binary
    global rtr_interp, current_MPBH
    if rtr_interp is not None and current_MPBH == M_PBH:
        return rtr_interp

    print(f"   Tabulating truncation radius r_tr(a) for M_total = {M_PBH} M_sun...")
    current_MPBH = M_PBH

    f_ref = 0.01 # Use f=0.01 as a reference for interpolation range
    am = a_max(f_ref, M_PBH, withHalo=True)
    a_list = np.logspace(-8, np.log10(am*1.1), 101)

    # Iterative calculation for M_halo(z_dec(a))
    z_decoupling_0 = z_decoupling(a_list, f_ref, M_PBH)
    M_halo_0 = M_halo(z_decoupling_0, M_PBH)

    z_decoupling_1 = np.zeros(len(a_list))
    M_halo_1 = np.zeros(len(a_list))
    for i in range(len(a_list)):
        z_decoupling_1[i] = z_decoupling(a_list[i], f_ref, (M_halo_0[i]+M_PBH))
        M_halo_1[i] = M_halo(z_decoupling_1[i], (M_PBH))

    z_decoupling_2 = np.zeros(len(a_list))
    M_halo_2 = np.zeros(len(a_list))
    for i in range(len(a_list)):
        z_decoupling_2[i] = z_decoupling(a_list[i], f_ref, (M_halo_1[i]+M_PBH))
        M_halo_2[i] = M_halo(z_decoupling_2[i], (M_PBH))

    z_decoupling_3 = np.zeros(len(a_list))
    for i in range(len(a_list)):
        z_decoupling_3[i] = z_decoupling(a_list[i], f_ref, (M_halo_2[i]+M_PBH))

    r_list = r_trunc(z_decoupling_3, M_PBH)
    rtr_interp = interp1d(a_list, r_list, bounds_error=False, fill_value="extrapolate")
    return rtr_interp

def rho(r, r_tr, M_PBH, gamma=3.0/2.0):
    x = r/r_tr
    if r_tr <= 0 or r_eq(M_PBH) <=0: return 0.0
    A_denom = (4*np.pi*(r_tr**gamma)*(r_eq(M_PBH)**(3-gamma)))
    if A_denom == 0: return 0.0
    A = (3-gamma)*M_PBH / A_denom
    if (x <= 1):
        return A*x**(-gamma)
    else:
        return 0

def Menc(r, r_tr, M_PBH, gamma=3.0/2.0):
    x = r/r_tr
    if r_eq(M_PBH) <= 0: return M_PBH
    if (x <= 1):
        return M_PBH*(1+(r/r_eq(M_PBH))**(3-gamma))
    else:
        return M_PBH*(1+(r_tr/r_eq(M_PBH))**(3-gamma))

def calcBindingEnergy(r_tr, M_PBH):
    if r_tr <= 1e-8: return 0.0
    integ = lambda r: Menc(r, r_tr, M_PBH)*rho(r, r_tr, M_PBH)*r
    return -G_N*4*np.pi*quad(integ,1e-8, 1.0*r_tr, epsrel=1e-3)[0]

def getBindingEnergy(r_tr, M_PBH):
    global current_MPBH, Ubind_interp, rtr_interp
    if ((M_PBH - current_MPBH)**2 > 1e-3 or Ubind_interp is None):
        print(f"   Tabulating binding energy U_bind(r_tr) for M_total = {M_PBH} M_sun...")
        current_MPBH = M_PBH
        rtr_vals = np.logspace(np.log10(1e-8), np.log10(1.0*r_eq(M_PBH)), 500)
        Ubind_vals = np.asarray([calcBindingEnergy(r1, M_PBH) for r1 in rtr_vals])
        Ubind_interp = interp1d(rtr_vals, Ubind_vals, bounds_error=False, fill_value="extrapolate")

        rtr_interp = GetRtrInterp(M_PBH)

    return Ubind_interp(r_tr)

# --- Analytical PDF Functions (from MergerRate.ipynb) ---
def j_X(x, f, M_PBH):
    if f <= 0: f = 1e-9
    return bigX(x, f, M_PBH)*0.5*(1+sigma_eq**2/(0.85*f)**2)**0.5

def P_j(j, x, f, M_PBH):
    jX = j_X(x, f, M_PBH)
    if j <= 0 or jX <= 0: return 0.0
    y = j/jX
    return (y**2/(1+y**2)**(3.0/2.0))/j

def P_a_j_withHalo(a, j, f, M_PBH):
    # This is the PDF we sample from: the *initial* distribution including the halo mass
    # M_PBH is the total mass of the binary
    xval = x_of_a(a, f, M_PBH)
    if xval <= 0 or a <= 0: return 0.0

    X = bigX(xval, f, M_PBH)
    xb = xbar(f, M_PBH)
    if xb <= 0: return 0.0

    measure = (3.0/4.0)*(a**-0.25)*(0.85*f/(alpha*xb))**0.75
    z_dec = z_decoupling(a, f, M_PBH)
    M_h = M_halo(z_dec, M_PBH)
    measure *= ((M_PBH + M_h)/M_PBH )**(3./4.)

    prob_j = P_j(j, xval, f, M_PBH)
    if prob_j <= 0 or not np.isfinite(prob_j): return 0.0

    return prob_j * np.exp(-X) * measure

def P_la_lj_withHalo(la, lj, f, M_PBH):
    # Logarithmic probability for MCMC
    j = 10**lj
    a = 10**la
    return P_a_j_withHalo(a, j, f, M_PBH)*a*j*(np.log(10)**2)

# --- Coalescence and Remapping Functions (from Remapping.py & Sampling.py) ---
def t_coal(a, e, M_total, eta):
    # Units: a [pc], M_total [M_sun], returns t [years]
    # This function now correctly includes eta

    # CRASH FIX: Detect if 'a' is scalar or array
    is_scalar = np.isscalar(a)

    # Ensure inputs are arrays for vectorized operations
    a_arr = np.atleast_1d(a)
    e_arr = np.atleast_1d(e)

    # Create a result array, set invalid entries to 1e99
    t = np.full_like(a_arr, 1e99)

    # Find valid entries
    valid_mask = (a_arr > 0) & (e_arr >= 0) & (e_arr < 1) & (eta > 0) & (M_total > 0)

    if not np.any(valid_mask):
        return t[0] if is_scalar else t # Return scalar if input was scalar

    a_valid = a_arr[valid_mask]
    e_valid = e_arr[valid_mask]

    # Original Q from GitHub code assumed m1=m2, eta=0.25
    # Paper Eq. 7: tau ~ 1 / (eta * M^3)
    # We must scale Q by (0.25 / eta)

    Q_base = (3.0/170.0)*(G_N*M_total)**(-3) # This is Q for eta=0.25
    Q_corrected = Q_base * (0.25 / eta)   # Scale for arbitrary eta

    j_sq_valid = 1.0 - e_valid**2
    # Prevent j_sq = 0 which would lead to inf/nan
    j_sq_valid = np.clip(j_sq_valid, 1e-300, 1.0)

    tc_valid = Q_corrected * a_valid**4 * (j_sq_valid)**(3.5) #s^6 pc km^-6
    tc_valid *= 3.086e+13 #s^6 km^-5
    tc_valid *= (3e5)**5 #s

    t[valid_mask] = tc_valid/(60*60*24*365) #in years

    # CRASH FIX: Return scalar if input was scalar, otherwise return array
    return t[0] if is_scalar else t


def j_coal(a, t, M_total, eta):
    # Inverse of t_coal, gives j for a given t
    if a <= 0 or t <= 0 or eta <= 0 or M_total <= 0: return 1.0 # Return invalid j

    Q_base = (3.0/170.0)*(G_N*M_total)**-3
    Q_corrected = Q_base * (0.25 / eta)

    tc = t*(60*60*24*365)
    tc /= (3e5)**5
    tc /= 3.086e+13

    j_sq_pow_7 = (tc/(Q_corrected*a**4))
    if j_sq_pow_7 < 0: return 1.0

    j_sq = j_sq_pow_7**(2.0/7.0)
    return np.sqrt(j_sq) # Returns j

def calc_af(ai, M_total):
    # Calculates remapped final semi-major axis
    global current_MPBH, rtr_interp, Ubind_interp

    # ai can be an array, need to handle initialization
    if ((M_total - current_MPBH)**2 > 1e-3 or rtr_interp is None or Ubind_interp is None):
        getBindingEnergy(1e-5, M_total) # This initializes all interpolators

    r_tr = rtr_interp(ai)
    r_tr = np.clip(r_tr, 1e-9, None) # Ensure r_tr is positive

    Mtot_halo = Menc(r_tr, r_tr, M_total)
    U_orb_before = -G_N*(Mtot_halo**2)/(2.0*ai)

    # Use r_eq(M_total) for r_eq comparison
    r_eq_val = r_eq(M_total)

    # Vectorized check for r_tr > r_eq_val
    Ubind = np.zeros_like(r_tr)
    mask_large = r_tr > r_eq_val
    mask_small = ~mask_large

    if np.any(mask_large):
        Ubind[mask_large] = getBindingEnergy(r_eq_val, M_total) # Constant value
    if np.any(mask_small):
        Ubind[mask_small] = getBindingEnergy(r_tr[mask_small], M_total)

    U_final = U_orb_before + 2.0*Ubind

    # Calculate af, handle unbound cases
    af = np.full_like(ai, -1.0)
    bound_mask = (U_final < 0) & np.isfinite(U_final)

    af[bound_mask] = -G_N*M_total**2*0.5/(U_final[bound_mask]) # Use M_total (binary mass)

    return af

def calc_jf(ji, ai, af):
    # Calculates remapped final angular momentum
    # af, ai, ji are arrays
    jf = np.full_like(ji, -1.0)

    valid_mask = (af > 0) & (ai > 0)
    if np.any(valid_mask):
        jf[valid_mask] = ji[valid_mask] * np.sqrt(ai[valid_mask] / af[valid_mask])

    return jf


# --- 2. MCMC Sampling Functions (from Sampling.py) ---
tmin_sampling = 1e9 # 1 Gyr
tmax_sampling = 1e11 # 100 Gyr (older than universe, just a wide range)

def lnprior(theta, M_total, eta, a1, a2):
    la, lj = theta
    a = 10**la
    j = 10**lj

    if (j > 1 or j <= 0):
        return -np.inf

    if (a < a1 or a > a2):
        return -np.inf

    # lnprior is called with scalars, so this call is safe
    t = t_coal(a, np.sqrt(1-j**2), M_total, eta)
    if (t < tmin_sampling or t > tmax_sampling):
        return -np.inf

    return 0

def lnprob(theta, f, M_total, eta, PDF, a1, a2):
    lp = lnprior(theta, M_total, eta, a1, a2)
    if not np.isfinite(lp):
        return -np.inf

    la, lj = theta

    prob = PDF(la, lj, f, M_total)
    if prob <= 0 or not np.isfinite(prob):
        return -np.inf

    return lp + np.log(prob)

def GetSamples_MCMC(N_samps, PDF, a1, a2, f, M_total, eta):
    ndim, nwalkers = 2, 50 # Increased walkers

    # Initial guess for sampler
    a0 = np.sqrt(a1*a2)
    t_guess = 13.7e9 # Age of universe in years
    j0 = j_coal(a0, t_guess, M_total, eta)
    if j0 > 1.0 or j0 <= 0 or not np.isfinite(j0): j0 = 0.5 # Ensure valid starting j

    p0 = [[np.log10(a0), np.log10(j0)] + 1e-4*np.random.rand(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[f, M_total, eta, PDF, a1, a2])

    # Burn-in
    n_burn = 1000
    try:
        state = sampler.run_mcmc(p0, n_burn, progress=True)
    except Exception as e:
        print(f"Error during MCMC burn-in: {e}")
        print(f"Initial parameters: a0={a0}, j0={j0}")
        return np.array([])

    sampler.reset()

    # Main run
    n_run = int(N_samps / nwalkers)
    try:
        sampler.run_mcmc(state, n_run, progress=True)
    except Exception as e:
        print(f"Error during MCMC main run: {e}")
        return np.array([])

    samples = sampler.get_chain(flat=True)
    print(f"   Generated {len(samples)} samples...")
    return samples

# --- 3. Main Simulation Function ---
def run_statistical_simulation(cfg):
    start_time = time.time()

    # --- Load Config ---
    try:
        f_pbh = cfg.pbh_population.f_pbh
        m1_solar = float(cfg.pbh_population.m1_solar) # Use new key
        m2_solar = float(cfg.pbh_population.m2_solar) # Use new key
        output_dir = cfg.output.save_path
        N_samples = int(cfg.stage1.num_particles) # Use this for sample count
    except AttributeError as e:
        print(f"❌ Configuration Error: Missing key in YAML file: {e}")
        return
    except ValueError as e:
        print(f"❌ Configuration Error: Invalid mass value in YAML file: {e}")
        return

    # Calculate binary properties
    M_total_solar = m1_solar + m2_solar
    eta = (m1_solar * m2_solar) / (M_total_solar**2)

    print(f"✅ Statistical simulation starting.")
    print(f"   f_pbH = {f_pbh}")
    print(f"   m1 = {m1_solar} M_sun, m2 = {m2_solar} M_sun")
    print(f"   M_total = {M_total_solar} M_sun, eta = {eta:.4f}")

    # --- Initialize Interpolators ---
    global current_MPBH
    current_MPBH = -10.0 # Force re-tabulation
    amin = 5e-5
    amax = a_max(f_pbh, M_total_solar, withHalo=True) # Use remapped a_max

    _ = GetRtrInterp(M_total_solar)
    _ = getBindingEnergy(1e-5, M_total_solar)

    # --- Generate Samples ---
    print(f"Sampling {N_samples} binaries from analytical PDF...")
    samples = GetSamples_MCMC(N_samples, P_la_lj_withHalo, amin, amax, f_pbh, M_total_solar, eta)

    if samples.size == 0:
        print("❌ MCMC sampling failed. Exiting.")
        return

    la_samps = samples[:,0]
    lj_samps = samples[:,1]

    a_initial = 10**la_samps
    j_initial = 10**lj_samps
    e_initial = np.sqrt(1 - j_initial**2)

    # Calculate initial merger times (before remapping)
    # This call now uses arrays, which t_coal can handle
    t_initial_yrs = t_coal(a_initial, e_initial, M_total_solar, eta)

    # --- Remap Samples ---
    print("Remapping samples with 'dark dress' physics...")

    # Vectorized calculation of af and jf
    a_final = calc_af(a_initial, M_total_solar)
    j_final = calc_jf(j_initial, a_initial, a_final)

    # Filter out unbound binaries
    bound_mask = (a_final > 0) & (j_final >= 0) & (j_final <= 1.0)
    a_final_bound = a_final[bound_mask]
    j_final_bound = j_final[bound_mask]

    # Ensure eccentricity is non-negative
    e_final_bound_sq = 1 - j_final_bound**2
    e_final_bound = np.sqrt(np.clip(e_final_bound_sq, 0, 1.0))

    # Calculate final merger times
    t_final_yrs = t_coal(a_final_bound, e_final_bound, M_total_solar, eta)

    print(f"   Initial state: {len(a_initial)} samples.")
    print(f"   Final state: {len(a_final_bound)} bound binaries.")

    # --- Save Results ---
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f'statistical_merger_times_M_total_{M_total_solar}_f{f_pbh:.0e}.npz')

    np.savez_compressed(
        output_filename,
        t_initial_yrs=t_initial_yrs,
        t_final_yrs=t_final_yrs,
        a_initial=a_initial,
        j_initial=j_initial,
        a_final_bound=a_final_bound,
        j_final_bound=j_final_bound,
        f_pbh=f_pbh,
        m1_solar=m1_solar,
        m2_solar=m2_solar,
        M_total_solar=M_total_solar,
        eta=eta
    )

    print(f"\n✅ Statistical simulation complete. Results saved to '{output_filename}'")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Statistical (MCMC) PBH merger simulation.")
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    cfg = load_config(args.config_file)

    # Add keys if they don't exist (for compatibility with our old config)
    if not hasattr(cfg, 'stage1'):
        cfg.stage1 = type('',(object,),{'num_particles': 10000})() # 10k samples for a test
    if not hasattr(cfg.pbh_population, 'f_pbh'):
         cfg.pbh_population.f_pbh = 0.1
    # Check for new mass keys, fallback to old one
    if not hasattr(cfg.pbh_population, 'm1_solar'):
         cfg.pbh_population.m1_solar = float(cfg.pbh_population.f_pbh_mu)
    if not hasattr(cfg.pbh_population, 'm2_solar'):
         cfg.pbh_population.m2_solar = float(cfg.pbh_population.f_pbh_mu)

    if not hasattr(cfg.pbh_population, 'f_pbh_sigma'):
        cfg.pbh_population.f_pbh_sigma = 0.5
    if not hasattr(cfg, 'output'):
        cfg.output = type('',(object,),{'save_path': 'results_statistical_sim/'})

    run_statistical_simulation(cfg)

