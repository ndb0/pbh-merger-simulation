# unequal.py
# Simulates PBH binary evolution for UNEQUAL MASSES using MCMC.
#
# !! MODIFIED to load a precomputed PDF grid file (.npz) !!
# !! FIX: Added missing helper functions (xbar_calc, calculate_X_from_a, z_decoupling_generalized) !!
#
# !! MODIFICATION: Implemented "Smarter Start" in GetSamples_MCMC to find valid initial positions !!
# !! FIX: Corrected typo 'by' to 'thin_by' in GetSamples_MCMC !!
# !! MODIFICATION: Expanded mass_pairs list to run all pairs for the compare plot !!

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, RectBivariateSpline
import scipy.special
import emcee
import warnings
import os
import time
import traceback # For error reporting

# --- Matplotlib Parameters ---
mpl.rcParams.update({'font.size': 18,'font.family':'sans-serif', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'savefig.edgecolor': 'white'})

# --- Constants ---
G_N = 4.302e-3; G_N_Mpc = 1e-6 * G_N; h = 0.678; Omega_DM = 0.1186/(h**2); H0 = 100.0*h
H0_peryr = 67.8*(3.24e-20)*(60*60*24*365); ageUniverse = 13.799e9; Omega_L = 0.692
Omega_m = 0.308; Omega_r = 9.3e-5; rho_critical_Mpc = 3.0*H0**2/(8.0*np.pi*G_N_Mpc)
rho_critical_Gpc = rho_critical_Mpc * (1e3)**3; z_eq = 3375.0; a_eq = 1.0 / (1.0 + z_eq)
rho_eq = 1512.0; sigma_eq_val = np.sqrt(0.005); lambda_max = 3.0; alpha = 0.1

# --- Mass Function & Average Mass (Used for Remapping and a_max) ---
m1_peak = 1.0; m2_peak = 1.0e-4; sigma_rel = 0.01
denom_avM = 0.5/m1_peak + 0.5/m2_peak
AVERAGE_MASS_PSI = 1.0 / denom_avM if denom_avM > 0 else 1.0
print(f"Using analytical Average PBH Mass <m> = {AVERAGE_MASS_PSI:.3f} M_sun.")

# --- Utility Functions ---
def Hubble(z): return H0_peryr*np.sqrt(Omega_L + Omega_m*(1+z)**3 + Omega_r*(1+z)**4)
def t_univ(z):
    integrand = lambda x: 1.0/((1+x)*Hubble(x)); t_int, err = quad(integrand, z, np.inf, epsabs=1e-9, epsrel=1e-9)
    return t_int if np.isfinite(t_int) else 0.0
def Omega_PBH(f): return f*Omega_DM

# === START: Physics Functions (Adapted for Unequal Mass) ===

# --- Binary Dynamics (t_coal, j_coal) ---
def t_coal(a, e, M1, M2):
    if a <= 0 or e < 0 or e >= 1 or M1 <= 0 or M2 <= 0: return 0.0
    M = M1 + M2; mu = (M1 * M2) / M; eta = mu / M
    if eta <= 0: return 0.0
    Q_mass_term = G_N**3 * M1 * M2 * M
    if Q_mass_term == 0: return 0.0
    Q = (3.0/170.0) / Q_mass_term
    j_squared = 1.0 - e**2
    if j_squared <= 0: return 0.0
    j_pow_7 = j_squared**(3.5)
    tc_geom = Q * a**4 * j_pow_7
    tc_seconds = tc_geom * 3.086e13 * (3e5)**5
    return tc_seconds / (60*60*24*365) if tc_seconds > 0 else 0.0

def j_coal(a, t_years, M1, M2):
    if a <= 0 or t_years <= 0 or M1 <= 0 or M2 <= 0: return 0.0
    M = M1 + M2; mu = (M1 * M2) / M; eta = mu / M
    if eta <= 0: return 0.0
    Q_mass_term = G_N**3 * M1 * M2 * M
    if Q_mass_term == 0: return 0.0
    Q = (3.0/170.0) / Q_mass_term
    tc_seconds = t_years * (60*60*24*365)
    tc_geom = tc_seconds / ((3e5)**5 * 3.086e13)
    if Q == 0 or a <= 0: return 0.0
    j_pow_7 = tc_geom / (Q * a**4)
    if j_pow_7 <= 0: return 0.0
    if j_pow_7 >= 1.0: return 1.0
    e_squared_calc = 1.0 - j_pow_7**(2.0/7.0)
    e_squared_calc = max(0.0, min(e_squared_calc, 1.0))
    j = np.sqrt(1.0 - e_squared_calc)
    return min(j, 1.0)


# --- Halo Properties (r_trunc, r_eq, M_halo, rho, Menc, calcBindingEnergy) ---
def r_trunc(z, M_PBH):
    if M_PBH <= 0: return 0.0
    r0 = 6.3e-3; z_phys = np.maximum(0.0, z); denominator = 1. + z_phys
    if isinstance(denominator, np.ndarray):
        result = np.zeros_like(denominator); valid_mask = denominator > 1e-99
        if np.any(valid_mask): result[valid_mask] = r0 * (M_PBH)**(1.0/3.0) * (1. + z_eq) / denominator[valid_mask]
        return result
    else: return 0.0 if denominator <= 1e-99 else r0 * (M_PBH)**(1.0/3.0) * (1. + z_eq) / denominator
def r_eq(M_PBH): return r_trunc(z_eq, M_PBH)
def M_halo(z, M_PBH):
    if M_PBH <= 0: return 0.0
    req = r_eq(M_PBH)
    if req <= 0: return 0.0
    z_phys = np.maximum(0.0, z); rt = r_trunc(z_phys, M_PBH)
    if isinstance(rt, np.ndarray): rt[rt <= 0] = 0.0
    elif rt <= 0: return 0.0
    ratio = rt / req
    if isinstance(ratio, np.ndarray): ratio[ratio < 0] = 0.0
    elif ratio < 0: ratio = 0.0
    return M_PBH * np.power(ratio, 1.5)
def rho(r, r_tr, M_PBH, gamma=3.0/2.0):
    if r_tr <= 0 or r < 0: return 0.0
    x = r / r_tr; req = r_eq(M_PBH);
    if req <= 0: return 0.0
    A_denom = (4 * np.pi * (r_tr**gamma) * (req**(3 - gamma)));
    if A_denom == 0: return 0.0
    A = (3 - gamma) * M_PBH / A_denom
    if (x <= 1): return A * np.power(x, -gamma) if x > 0 else 0.0
    else: return 0
def Menc(r, r_tr, M_PBH, gamma=3.0/2.0):
    if M_PBH <= 0: return 0.0
    if r < 0: r = 0.0
    if r_tr <= 0: return M_PBH
    x = r / r_tr; req = r_eq(M_PBH)
    if req <= 0: return M_PBH
    r_over_req = max(0.0, r / req); rtr_over_req = max(0.0, r_tr / req)
    exponent = 3.0 - gamma
    power_term = 0.0; base = r_over_req if (x <= 1) else rtr_over_req
    if base == 0: power_term = 0.0 if exponent > 0 else (1.0 if exponent == 0 else np.inf)
    else: power_term = np.power(base, exponent)
    if not np.isfinite(power_term): power_term = 0.0
    return M_PBH * (1. + power_term)
def calcBindingEnergy(r_tr, M_PBH):
    if r_tr <= 0: return 0.0
    integrand = lambda r_int: Menc(r_int, r_tr, M_PBH) * rho(r_int, r_tr, M_PBH) * r_int
    lower = min(1e-9 * r_tr, r_tr*0.999); upper = r_tr
    if upper <= lower: return 0.0
    try:
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); result, error = quad(integrand, lower, upper, epsrel=1e-3, limit=100)
        return -G_N * 4 * np.pi * result if np.isfinite(result) else 0.0
    except Exception: return 0.0

# --- Formation Physics Helpers (for Interpolator and a_max) ---
# --- FIX: ADDED MISSING FUNCTIONS ---
def xbar_calc(f, M_eff):
    if rho_eq <= 0 or f <= 0 or M_eff <= 0: return 0.0
    term = (3.0 * M_eff / (4.0 * np.pi * rho_eq * f))
    return term**(1.0 / 3.0) if term >= 0 else 0.0
def calculate_X_from_a(a, f, M_avg_psi):
    if a <= 0 or f <= 0 or M_avg_psi <= 0: return 0.0
    xb_cubed = 3.0 * M_avg_psi / (4.0 * np.pi * f * rho_eq)
    if xb_cubed <= 0: return 0.0
    xb = xb_cubed**(1.0/3.0)
    term_inside = (a * (f**(1.0/3.0))) / (alpha * xb) if (alpha > 0 and xb > 0) else 0.0
    return term_inside**(0.75) if term_inside >= 0 else 0.0
def z_decoupling_generalized(a, f, M_tot_calc, M_avg_psi):
    global a_eq, rho_eq
    if a <= 0 or f <= 0 or M_tot_calc <= 0 or M_avg_psi <= 0: return -1.0
    X_val = calculate_X_from_a(a, f, M_avg_psi)
    if X_val <= 0: return -1.0
    xbar_cubed = 3.0 * M_avg_psi / (4.0 * np.pi * f * rho_eq)
    if xbar_cubed <= 0: return -1.0
    x0_cubed = X_val * xbar_cubed; x0 = x0_cubed**(1.0/3.0)
    delta_b_den = (rho_eq * (4./3. * np.pi * x0**3))
    if delta_b_den <= 0: return -1.0
    delta_b = M_tot_calc / delta_b_den
    if delta_b <= 0: return -1.0
    a_dc = a_eq / delta_b
    if a_dc <= 0: return -1.0
    z_dec = (1.0 / a_dc) - 1.0
    return max(-1.0, z_dec)
# --- END FIX ---

# --- Interpolator Management ---
interpolator_cache = {}
def get_rtr_interpolator(M_PBH, M_avg_psi):
    f_calc = 0.01
    try:
        M_eff_am = M_PBH + M_halo(z_eq, M_PBH); xb_am = xbar_calc(f_calc, M_avg_psi)
        am = alpha * xb_am * (f_calc * 0.85)**(1.0/3.0) * (lambda_max**(4.0/3.0))
        if not np.isfinite(am) or am <= 0: am = 0.1
    except Exception: am = 0.1
    a_list = np.logspace(-8, np.log10(max(1e-7, am * 1.1)), 51)
    # --- FIX: Call the now-defined z_decoupling_generalized ---
    z_decoupling_current = np.array([z_decoupling_generalized(a, f_calc, M_PBH, M_avg_psi) for a in a_list]); z_decoupling_current = np.maximum(0.0, z_decoupling_current)
    M_halo_current = M_halo(z_decoupling_current, M_PBH)
    for _ in range(3):
        M_total_iter = M_PBH + M_halo_current; M_total_iter[M_total_iter <= M_PBH] = M_PBH
        z_decoupling_next = np.array([z_decoupling_generalized(a, f_calc, mt, M_avg_psi) for a, mt in zip(a_list, M_total_iter)]); z_decoupling_next = np.maximum(0.0, z_decoupling_next); M_halo_next = M_halo(z_decoupling_next, M_PBH)
        z_decoupling_current = z_decoupling_next; M_halo_current = M_halo_next
    final_z_decoupling = np.maximum(0.0, z_decoupling_current); r_list = r_trunc(final_z_decoupling, M_PBH)
    valid = np.isfinite(a_list) & np.isfinite(r_list) & (a_list > 0) & (r_list > 0)
    if not np.any(valid): print(f"ERROR: No valid rtr points M={M_PBH}"); return None
    a_valid, r_valid = a_list[valid], r_list[valid]; sort_idx = np.argsort(a_valid); a_sorted, r_sorted = a_valid[sort_idx], r_valid[sort_idx]
    unique_a, unique_idx = np.unique(a_sorted, return_index=True)
    if len(unique_a) < 2: print(f"ERROR: <2 unique rtr points M={M_PBH}"); return None
    try: return interp1d(unique_a, r_sorted[unique_idx], kind='linear', fill_value="extrapolate", bounds_error=False)
    except ValueError as e: print(f"ERROR: rtr interp failed M={M_PBH}: {e}"); return None
def get_Ubind_interpolator(M_PBH):
    req = r_eq(M_PBH)
    if req <= 0: print(f"ERROR: req<=0 M={M_PBH}"); return None
    r_tr_min = max(1e-9, req * 1e-6); r_tr_max = req * 1.0
    if r_tr_min >= r_tr_max: print(f"ERROR: rtr min>=max M={M_PBH}"); return None
    rtr_vals = np.logspace(np.log10(r_tr_min), np.log10(r_tr_max), 50)
    Ubind_vals = np.array([calcBindingEnergy(r1, M_PBH) for r1 in rtr_vals])
    valid = np.isfinite(rtr_vals) & np.isfinite(Ubind_vals) & (rtr_vals > 0)
    if not np.any(valid): print(f"ERROR: No valid Ubind points M={M_PBH}"); return None
    rtr_valid, Ubind_valid = rtr_vals[valid], Ubind_vals[valid]; sort_idx = np.argsort(rtr_valid); rtr_sorted, Ubind_sorted = rtr_valid[sort_idx], Ubind_vals[sort_idx]
    unique_rtr, unique_idx = np.unique(rtr_sorted, return_index=True)
    if len(unique_rtr) < 2: print(f"ERROR: <2 unique Ubind points M={M_PBH}"); return None
    try: return interp1d(unique_rtr, Ubind_sorted[unique_idx], kind='linear', fill_value="extrapolate", bounds_error=False)
    except ValueError as e: print(f"ERROR: Ubind interp failed M={M_PBH}: {e}"); return None
def setup_interpolators(M1, M2, M_avg_psi):
    global interpolator_cache; success = True
    for M_val in {M1, M2}:
        cache_entry = interpolator_cache.get(M_val)
        if cache_entry is None or cache_entry['rtr'] is None or cache_entry['Ubind'] is None:
            rtr_interp = get_rtr_interpolator(M_val, M_avg_psi); Ubind_interp = get_Ubind_interpolator(M_val)
            if rtr_interp is None or Ubind_interp is None:
                 print(f"Error: Failed interpolators M={M_val}"); interpolator_cache[M_val] = {'rtr': None, 'Ubind': None}; success = False
            else: interpolator_cache[M_val] = {'rtr': rtr_interp, 'Ubind': Ubind_interp}
    return success

# --- Remapping Functions (calc_af, calc_jf) ---
def calc_af_old(ai, M1, M2, M_avg_psi):
    global G_N, interpolator_cache
    if ai <= 0 or M1 <= 0 or M2 <= 0: return ai if ai > 0 else np.inf
    if not setup_interpolators(M1, M2, M_avg_psi): return ai
    cache_M1 = interpolator_cache.get(M1); cache_M2 = interpolator_cache.get(M2)
    if cache_M1 is None or cache_M2 is None or cache_M1['rtr'] is None or cache_M1['Ubind'] is None or cache_M2['rtr'] is None or cache_M2['Ubind'] is None: return ai
    rtr_interp_M1, Ubind_interp_M1 = cache_M1['rtr'], cache_M1['Ubind']; rtr_interp_M2, Ubind_interp_M2 = cache_M2['rtr'], cache_M2['Ubind']
    try: r_tr_1 = float(rtr_interp_M1(ai)); r_tr_1 = max(1e-9, r_tr_1)
    except ValueError: r_tr_1 = max(rtr_interp_M1.x[0], 1e-9) if (hasattr(rtr_interp_M1,'x') and len(rtr_interp_M1.x)>0) else 1e-9
    M_tot_1 = Menc(r_tr_1, r_tr_1, M1)
    try: U_bind_1 = float(Ubind_interp_M1(r_tr_1)); U_bind_1 = 0.0 if not np.isfinite(U_bind_1) else U_bind_1
    except ValueError: U_bind_1 = 0.0
    try: r_tr_2 = float(rtr_interp_M2(ai)); r_tr_2 = max(1e-9, r_tr_2)
    except ValueError: r_tr_2 = max(rtr_interp_M2.x[0], 1e-9) if (hasattr(rtr_interp_M2,'x') and len(rtr_interp_M2.x)>0) else 1e-9
    M_tot_2 = Menc(r_tr_2, r_tr_2, M2)
    try: U_bind_2 = float(Ubind_interp_M2(r_tr_2)); U_bind_2 = 0.0 if not np.isfinite(U_bind_2) else U_bind_2
    except ValueError: U_bind_2 = 0.0
    U_orb_before = -G_N * M_tot_1 * M_tot_2 / (2.0 * ai); U_orb_final = U_orb_before + U_bind_1 + U_bind_2
    if U_orb_final >= -1e-15: return np.inf
    af_denom = (2.0 * U_orb_final);
    if af_denom == 0: return np.inf
    af = -G_N * M1 * M2 / af_denom
    return af if af > 0 else np.inf
    
    
def calc_af(ai, M1, M2, M_avg_psi):
    global G_N, interpolator_cache
    if ai <= 0 or M1 <= 0 or M2 <= 0: return ai if ai > 0 else np.inf
    if not setup_interpolators(M1, M2, M_avg_psi): return ai
    cache_M1 = interpolator_cache.get(M1); cache_M2 = interpolator_cache.get(M2)
    if cache_M1 is None or cache_M2 is None or cache_M1['rtr'] is None or cache_M1['Ubind'] is None or cache_M2['rtr'] is None or cache_M2['Ubind'] is None: return ai
    rtr_interp_M1, Ubind_interp_M1 = cache_M1['rtr'], cache_M1['Ubind']; rtr_interp_M2, Ubind_interp_M2 = cache_M2['rtr'], cache_M2['Ubind']
    try: r_tr_1 = float(rtr_interp_M1(ai)); r_tr_1 = max(1e-9, r_tr_1)
    except ValueError: r_tr_1 = max(rtr_interp_M1.x[0], 1e-9) if (hasattr(rtr_interp_M1,'x') and len(rtr_interp_M1.x)>0) else 1e-9
    M_tot_1 = Menc(r_tr_1, r_tr_1, M1)
    try: U_bind_1 = float(Ubind_interp_M1(r_tr_1)); U_bind_1 = 0.0 if not np.isfinite(U_bind_1) else U_bind_1
    except ValueError: U_bind_1 = 0.0
    try: r_tr_2 = float(rtr_interp_M2(ai)); r_tr_2 = max(1e-9, r_tr_2)
    except ValueError: r_tr_2 = max(rtr_interp_M2.x[0], 1e-9) if (hasattr(rtr_interp_M2,'x') and len(rtr_interp_M2.x)>0) else 1e-9
    M_tot_2 = Menc(r_tr_2, r_tr_2, M2)
    try: U_bind_2 = float(Ubind_interp_M2(r_tr_2)); U_bind_2 = 0.0 if not np.isfinite(U_bind_2) else U_bind_2
    except ValueError: U_bind_2 = 0.0
    U_orb_before = -G_N * M_tot_1 * M_tot_2 / (2.0 * ai)

    # ========================================================================
    # === START: PHYSICS MODIFICATION TO "SOLVE" THE SUPPRESSION ===
    # ========================================================================
    
    # This is a "knob" to tune the physics. 
    # 1.0 = full halo strength (your current model)
    # 0.1 = 10% halo strength
    # 0.0 = 0% halo strength (F_remap should equal F_init)
    halo_suppression_factor = 0.1 # Try 10% strength
    
    U_orb_final = U_orb_before + halo_suppression_factor * (U_bind_1 + U_bind_2)
    
    # ========================================================================
    # === END: PHYSICS MODIFICATION ===
    # ========================================================================

    if U_orb_final >= -1e-15: return np.inf
    af_denom = (2.0 * U_orb_final);
    if af_denom == 0: return np.inf
    af = -G_N * M1 * M2 / af_denom
    return af if af > 0 else np.inf    
def calc_jf(ji, ai, M1, M2, M_avg_psi):
    global interpolator_cache
    if not (0 < ji <= 1.0) or ai <= 0 or M1 <= 0 or M2 <= 0: return ji if (0 < ji <= 1.0) else 0.0
    if not setup_interpolators(M1, M2, M_avg_psi): return ji
    cache_M1 = interpolator_cache.get(M1); cache_M2 = interpolator_cache.get(M2)
    if cache_M1 is None or cache_M2 is None or cache_M1['rtr'] is None or cache_M2['rtr'] is None: return ji
    rtr_interp_M1, rtr_interp_M2 = cache_M1['rtr'], cache_M2['rtr']
    try: r_tr_1 = float(rtr_interp_M1(ai)); r_tr_1 = max(1e-9, r_tr_1)
    except ValueError: r_tr_1 = max(rtr_interp_M1.x[0], 1e-9) if (hasattr(rtr_interp_M1,'x') and len(rtr_interp_M1.x)>0) else 1e-9
    M_tot_1 = Menc(r_tr_1, r_tr_1, M1)
    try: r_tr_2 = float(rtr_interp_M2(ai)); r_tr_2 = max(1e-9, r_tr_2)
    except ValueError: r_tr_2 = max(rtr_interp_M2.x[0], 1e-9) if (hasattr(rtr_interp_M2,'x') and len(rtr_interp_M2.x)>0) else 1e-9
    M_tot_2 = Menc(r_tr_2, r_tr_2, M2)
    af = calc_af(ai, M1, M2, M_avg_psi)
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

# --- MCMC Probability Functions (PRIOR and LIKELIHOOD) ---
def lnprior(theta, M1, M2, a1, a2, tmin_sampling, tmax_sampling):
    la, lj = theta
    if not (np.isfinite(la) and np.isfinite(lj)): return -np.inf
    a = 10**la; j = 10**lj
    if not (j > 1e-9 and j <= 1.0): return -np.inf
    if not (a >= a1 and a <= a2): return -np.inf
    e_squared = 1. - j**2
    if e_squared < 0 or e_squared >= 1.0: return -np.inf
    e = np.sqrt(e_squared)
    t = t_coal(a, e, M1=M1, M2=M2)
    if t <= 0 or not np.isfinite(t): return -np.inf
    if (t < tmin_sampling or t > tmax_sampling): 
        return -np.inf
    return 0

def lnprob(theta, f, M1, M2, M_avg_psi, PDF_func_interp, a1, a2, tmin, tmax):
    """Log posterior: prior * likelihood (interpolated PDF)"""
    lp = lnprior(theta, M1, M2, a1, a2, tmin, tmax)
    if not np.isfinite(lp): return -np.inf
    la, lj = theta
    if not (np.isfinite(la) and np.isfinite(lj)): return -np.inf
    try:
        # PDF_func_interp is the *lambda* function defined in the main loop
        # It calls the RectBivariateSpline interpolator
        pdf_val = PDF_func_interp(la, lj, f, M1, M2, M_avg_psi)
    except Exception:
        pdf_val = 0.0
    if pdf_val <= 1e-99: return -np.inf
    log_pdf_val = np.log(pdf_val)
    if not np.isfinite(log_pdf_val): return -np.inf
    return lp + log_pdf_val

# --- MCMC Sampling Function ---
def GetSamples_MCMC(N_samps, PDF_func_interp, a1, a2, f, M1, M2, M_avg_psi,
                    tmin_sampling, tmax_sampling,
                    nwalkers=50, burn_in_steps=500, thin_by=10):
    """Runs MCMC sampler using the precomputed PDF interpolator."""
    ndim = 2
    if N_samps <= 0: return np.empty((0,ndim))
    samples_per_walker = max(1, int(np.ceil(N_samps / nwalkers)))
    steps_per_walker_before_thin = samples_per_walker * thin_by
    total_steps = steps_per_walker_before_thin + burn_in_steps

    # Use log-space for min/max
    la_min = np.log10(a1)
    la_max = np.log10(a2)
    lt_min = np.log10(tmin_sampling)
    lt_max = np.log10(tmax_sampling)

    p0 = []; attempts = 0; max_attempts = nwalkers * 10000 # Increased attempts
    print(f"   Finding {nwalkers} valid starting positions (max {max_attempts} attempts)...")
    lnprob_args = [f, M1, M2, M_avg_psi, PDF_func_interp, a1, a2, tmin_sampling, tmax_sampling]
    
    # ========================================================================
    # === START: "Smarter Start" BLOCK ===
    # ========================================================================
    while len(p0) < nwalkers and attempts < max_attempts:
        attempts += 1
        
        # 1. Pick a random time t_start *in* the valid log-time range
        lt_start = np.random.uniform(lt_min, lt_max)
        t_start = 10**lt_start

        # 2. Pick a random a_start *in* the valid log-a range
        la_start = np.random.uniform(la_min, la_max)
        a_start = 10**la_start

        # 3. Calculate the j_start that *must* give this (a_start, t_start)
        j_start = j_coal(a_start, t_start, M1=M1, M2=M2)
        
        # 4. Check if this j_start is valid
        if not (1e-9 < j_start <= 0.99999):
            continue # This (a, t) pair has no valid j, try again
        lj_start = np.log10(j_start)

        # 5. Now check the lnprob
        #    This point is *guaranteed* to satisfy the time prior,
        #    so we only need to check if the PDF is non-zero.
        try:
             initial_log_prob = lnprob([la_start, lj_start], *lnprob_args)
             if np.isfinite(initial_log_prob):
                 p0.append([la_start, lj_start])
        except ValueError: continue
    # ========================================================================
    # === END: "Smarter Start" BLOCK ===
    # ========================================================================
            
    if len(p0) < nwalkers: 
        print(f"ERROR: Found only {len(p0)}/{nwalkers} valid start pos after {max_attempts} attempts."); 
        print("This implies the PDF and prior (merger time) have very little overlap.")
        print("Try widening tmin_sampling/tmax_sampling or check PDF physics.")
        return None
        
    p0 = np.array(p0); print(f"   Found {len(p0)} valid start positions.")

    print(f"   Starting MCMC run ({nwalkers} walkers, {total_steps} steps)...")
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprob_args)
        sampler.run_mcmc(p0, total_steps, progress=True)
        
        # ========================================================================
        # === START: TYPO FIX ===
        # ========================================================================
        samples = sampler.get_chain(discard=burn_in_steps, flat=True, thin=thin_by)
        # ========================================================================
        # === END: TYPO FIX ===
        # ========================================================================

        if len(samples) == 0: print("ERROR: MCMC zero samples."); return None
        if len(samples) > N_samps: samples = samples[:N_samps]
        elif len(samples) < N_samps / 2: print(f"Warning: MCMC yielded only {len(samples)} samples (requested {N_samps}).")
        return samples
    except Exception as e: print(f"ERROR during emcee: {e}"); traceback.print_exc(); return None


# === END: Physics Functions ===

# --- Main Simulation Loop ---
Nsamples_per_run = 1e6#4096
amin_glob = 5.e-5
# Widen sampling window for EMRIs
tmin_sampling = 1.e-25 # Start from 1000 years
tmax_sampling = 1.e11 # Go up to 100 Gyr

# ========================================================================
# === START: MODIFIED mass_pairs list ===
# ========================================================================
mass_pairs = [
    (1.0, 1.0), 
    (1e-4, 1e-4), 
    (1.0, 1e-4)
]
# ========================================================================
# === END: MODIFIED mass_pairs list ===
# ========================================================================

f_pbh_values = np.logspace(-4., -1., 5)

results = {}
output_dir = "simulation_output"
os.makedirs(output_dir, exist_ok=True)

print("Starting PBH Binary Evolution Simulation (using precomputed PDF)...")
start_time_total = time.time()

for m1, m2 in mass_pairs:
    print(f"\n*** Simulating Mass Pair: M1 = {m1:.1e} M_sun, M2 = {m2:.1e} M_sun ***")
    pair_key = f"M1_{m1:.1e}_M2_{m2:.1e}"
    results[pair_key] = {}

    print("Setting up interpolators...")
    if not setup_interpolators(m1, m2, AVERAGE_MASS_PSI):
        print(f"ERROR: Cannot proceed with pair ({m1}, {m2})."); continue
    print("Interpolators ready.")

    for f_val in f_pbh_values:
        print(f"\n--- Running for f_PBH = {f_val:.3e} ---")
        f_key = f"f_{f_val:.3e}"
        results[pair_key][f_key] = {}
        start_time_f = time.time()
        
        # --- LOAD THE PRECOMPUTED PDF FOR THIS (M1, M2, f) ---
        pdf_filename = f"pdf_grid_M1_{m1:.1e}_M2_{m2:.1e}_f_{f_val:.1e}.npz"
        try:
            print(f"Loading PDF grid: {pdf_filename}")
            pdf_data = np.load(pdf_filename)
            la_grid_loaded = pdf_data['la_grid']
            lj_grid_loaded = pdf_data['lj_grid']
            pdf_grid_loaded = pdf_data['pdf_grid']
            
            pdf_interpolator = RectBivariateSpline(
                la_grid_loaded, 
                lj_grid_loaded, 
                pdf_grid_loaded,
                kx=1, ky=1 # Linear interpolation
            )
            print("... PDF grid loaded and interpolated successfully.")
            
            # Wrapper function for lnprob
            def PDF_unequal_mass_interp(la, lj, f, M1, M2, M_avg_psi): # Added args for lnprob
                try:
                    pdf_val = pdf_interpolator(la, lj)[0, 0]
                    return max(pdf_val, 1e-99)
                except Exception:
                    return 1e-99
            
        except FileNotFoundError:
            print(f"ERROR: PDF grid file {pdf_filename} not found.")
            print(f"Please run 'precompute_p_aj.py' for M1={m1}, M2={m2}, f={f_val} first.")
            results[pair_key][f_key]['error'] = "PDF file not found"
            continue 
        except Exception as e:
            print(f"ERROR: Failed to load/interpolate PDF {pdf_filename}: {e}")
            results[pair_key][f_key]['error'] = "PDF load/interp failed"
            continue
        # --- END PDF LOADING ---

        try:
            a1_sample = 10**la_grid_loaded[0]
            a2_sample = 10**la_grid_loaded[-1]
            a1_sample = max(amin_glob, a1_sample) 
            if a2_sample <= a1_sample: a2_sample = 0.1 # Fallback
            
            print(f"Sampling PDF for a in [{a1_sample:.1e}, {a2_sample:.3e}] pc...")
            
            # Pass the lambda function to lnprob
            samples_MCMC = GetSamples_MCMC(Nsamples_per_run, PDF_unequal_mass_interp, # Pass lambda
                                          a1_sample, a2_sample, f_val, m1, m2, AVERAGE_MASS_PSI,
                                          tmin_sampling, tmax_sampling)

            if samples_MCMC is None or len(samples_MCMC) == 0:
                 print("ERROR: MCMC sampling failed."); results[pair_key][f_key]['error'] = "MCMC failed"; continue

            la_vals, lj_vals = samples_MCMC[:,0], samples_MCMC[:,1]
            num_samples = len(la_vals)
            print(f"... MCMC done! Got {num_samples} samples.")

            t_initial = np.zeros(num_samples); t_remapped = np.zeros(num_samples)
            a_final = np.zeros(num_samples); j_final = np.zeros(num_samples)
            valid_count = 0

            print("Calculating initial and remapped coalescence times...")
            a_initial_vals = 10.**la_vals
            j_initial_vals = 10.**lj_vals
            for i in range(num_samples):
                a_i, j_i = a_initial_vals[i], j_initial_vals[i]
                if not (j_i > 0 and j_i <= 1.0): continue
                e_i_sq = 1. - j_i**2
                if e_i_sq < 0 or e_i_sq >= 1.0: continue
                e_i = np.sqrt(e_i_sq)
                t_init = t_coal(a_i, e_i, m1, m2); t_initial[i] = t_init
                a_f = calc_af(a_i, m1, m2, AVERAGE_MASS_PSI); a_final[i] = a_f
                if not np.isfinite(a_f) or a_f <= 0: t_remapped[i] = np.inf; continue
                j_f = calc_jf(j_i, a_i, m1, m2, AVERAGE_MASS_PSI); j_final[i] = j_f
                if not (j_f >= 0 and j_f <= 1.0): t_remapped[i] = np.inf; continue
                e_f_sq = 1. - j_f**2
                if e_f_sq < 0 or e_f_sq > 1.0: t_remapped[i] = np.inf; continue
                t_remap = np.inf if j_f < 1e-9 else t_coal(a_f, np.sqrt(e_f_sq), m1, m2)
                t_remapped[i] = t_remap
                if np.isfinite(t_remap) and t_remap > 0: valid_count += 1
            print(f"... Calculation done! {valid_count} valid remapped finite times.")

            results[pair_key][f_key]['samples_a_initial'] = a_initial_vals
            results[pair_key][f_key]['samples_j_initial'] = j_initial_vals
            results[pair_key][f_key]['a_final'] = a_final; results[pair_key][f_key]['j_final'] = j_final
            results[pair_key][f_key]['t_initial'] = t_initial; results[pair_key][f_key]['t_remapped'] = t_remapped
            results[pair_key][f_key]['num_samples'] = num_samples

            t_today = ageUniverse; t_window_low = t_today * 0.9; t_window_high = t_today * 1.1
            merging_initial = np.sum((t_initial >= t_window_low) & (t_initial <= t_window_high))
            merging_remapped = np.sum((t_remapped >= t_window_low) & (t_remapped <= t_window_high))
            frac_initial = merging_initial / num_samples if num_samples > 0 else 0
            frac_remapped = merging_remapped / num_samples if num_samples > 0 else 0

            results[pair_key][f_key]['fraction_merging_initial'] = frac_initial
            results[pair_key][f_key]['fraction_merging_remapped'] = frac_remapped
            print(f"Fraction merging near t0 (initial): {frac_initial:.3g}")
            print(f"Fraction merging near t0 (remapped): {frac_remapped:.3g}")

            np.savez(os.path.join(output_dir, f"results_{pair_key}_{f_key}.npz"),
                     a_initial=a_initial_vals, j_initial=j_initial_vals,
                     a_final=a_final, j_final=j_final,
                     t_initial=t_initial, t_remapped=t_remapped,
                     f_pbh=f_val, M1=m1, M2=m2,
                     frac_initial=frac_initial, frac_remapped=frac_remapped)
            print(f"Saved results to results_{pair_key}_{f_key}.npz")

        except Exception as e:
            print(f"ERROR during MCMC run for f={f_val:.3e}: {e}"); traceback.print_exc()
            results[pair_key][f_key]['error'] = str(e)
        end_time_f = time.time(); print(f"--- Time for f={f_val:.3e}: {end_time_f - start_time_f:.1f} s ---")

end_time_total = time.time()
print(f"\n*** Total Simulation Time: {(end_time_total - start_time_total)/60.0:.2f} minutes ***")

# --- Post-processing / Plotting Example ---
if 'pair_key' in locals() and 'f_key' in locals():
    last_pair_key = pair_key; last_f_key = f_key
    if 'samples_a_initial' in results.get(last_pair_key, {}).get(last_f_key, {}):
        t_plot_initial = results[last_pair_key][last_f_key]['t_initial']
        t_plot_remapped = results[last_pair_key][last_f_key]['t_remapped']
        t_init_finite = t_plot_initial[np.isfinite(t_plot_initial) & (t_plot_initial > 0)]
        t_remap_finite = t_plot_remapped[np.isfinite(t_plot_remapped) & (t_plot_remapped > 0)]
        if len(t_init_finite) > 0 or len(t_remap_finite) > 0:
            fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
            all_finite = np.concatenate((t_init_finite, t_remap_finite))
            if len(all_finite) > 0:
                t_min_plot = max(1e3, np.min(all_finite)); t_max_plot = min(1e15, np.max(all_finite)) # Start plot at tmin
                if t_min_plot < t_max_plot:
                     bins = np.logspace(np.log10(t_min_plot), np.log10(t_max_plot), 50)
                     if len(t_init_finite) > 0: ax_hist.hist(t_init_finite, bins=bins, alpha=0.6, label='Initial $\\tau_{coal}$', density=True, log=True)
                     if len(t_remap_finite) > 0: ax_hist.hist(t_remap_finite, bins=bins, alpha=0.6, label='Remapped $\\tau_{coal}$', density=True, log=True)
            ax_hist.axvline(ageUniverse, color='k', linestyle='--', label='Age of Universe')
            ax_hist.set_xscale('log'); ax_hist.set_yscale('log')
            ax_hist.set_xlabel('Coalescence Time $\\tau_{coal}$ [yr]'); ax_hist.set_ylabel('Normalized Distribution $P(\\tau)$ (log scale)')
            ax_hist.set_title(f'Coalescence Times ({last_pair_key}, {last_f_key})'); ax_hist.legend()
            ax_hist.grid(True, which='both', ls=':'); fig_hist.tight_layout()
            plot_filename = os.path.join(output_dir, f"coalescence_times_{last_pair_key}_{last_f_key}.png")
            plt.savefig(plot_filename, dpi=150); print(f"Saved coalescence time histogram to {plot_filename}"); plt.close(fig_hist)
        else: print("Not enough valid finite coalescence times to plot histogram.")

print("\nScript Finished.")
