# precompute_pdf.py
#
# Calculates the accurate semi-analytic P(log a, log j) distribution (Raidal et al. 2019, Eq. 2.31)
# FIX: Corrected ValueError in parallel worker function.
# FIX: Corrected 'mm2_peak' typo.
# FIX: Using robust analytical moments.
# IMPLEMENTED: CPU Parallelization via joblib.

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import scipy.stats
from joblib import Parallel, delayed # For multiprocessing
import time
import warnings
import os
import multiprocessing # To determine core count
import traceback

# --- Constants ---
G_N = 4.302e-3
rho_eq = 1512.0
alpha = 0.1
sigma_eq_val = np.sqrt(0.005)
z_eq = 3375.0
a_eq = 1.0 / (1.0 + z_eq)

# --- Mass Function Definition ---
m1_peak = 1.0; m2_peak = 1.0e-4; sigma_rel = 0.01
sigma_g1 = max(sigma_rel * m1_peak, 1e-12 * m1_peak)
sigma_g2 = max(sigma_rel * m2_peak, 1e-12 * m2_peak)
m_min_psi = m2_peak / 1000; m_max_psi = m1_peak * 10

def mass_func_psi_normalized(m):
    """Normalized bimodal Gaussian mass function psi(m), integral dm = 1."""
    pdf1 = 0.5 * scipy.stats.norm.pdf(m, loc=m1_peak, scale=sigma_g1)
    pdf2 = 0.5 * scipy.stats.norm.pdf(m, loc=m2_peak, scale=sigma_g2) 
    result = pdf1 + pdf2
    return np.maximum(0.0, result) if isinstance(m, np.ndarray) else max(0.0, result)

# --- CORRECTED Analytical Moment Calculation ---
def calculate_analytical_moments():
    """Calculates <m> and <m^2> analytically for the sharp bimodal case."""
    denom_avM = 0.5/m1_peak + 0.5/m2_peak
    avM = 1.0 / denom_avM if denom_avM > 0 else 1.0
    avM2 = 0.5 * m1_peak**2 + 0.5 * m2_peak**2
    return avM, avM2

# --- Physics Helper Functions (expect SCALAR inputs) ---
def calculate_X_from_a(a, f, M_avg_psi):
    """Calculates X from a scalar 'a'."""
    if a <= 0 or f <= 0 or M_avg_psi <= 0: return 0.0
    xb_cubed = 3.0 * M_avg_psi / (4.0 * np.pi * f * rho_eq)
    if xb_cubed <= 0: return 0.0
    xb = xb_cubed**(1.0/3.0)
    term_inside = (a * (f**(1.0/3.0))) / (alpha * xb) if (alpha > 0 and xb > 0) else 0.0
    return term_inside**(0.75) if term_inside >= 0 else 0.0

def calculate_j0(a, f, M1, M2, M_avg_psi, c_j=1.0):
    """Calculates j0 from a scalar 'a'."""
    M_total = M1 + M2
    if M_total <= 0 or M_avg_psi <=0: return 0.0
    X = calculate_X_from_a(a, f, M_avg_psi)
    if X <= 0: return 0.0
    j0 = c_j * X * M_avg_psi / M_total
    return max(j0, 1e-99)

def calculate_N_y(M1, M2, M_avg_psi, f, sigma_M):
    M_total = M1 + M2
    if M_avg_psi <= 0 or (f + sigma_M) <= 0: return 0.0
    return (M_total / M_avg_psi) * (f / (f + sigma_M))

def F_z_func(z):
    if z < 0: return 0.0 
    if z > 100: return z - 1
    z_sq = z**2
    arg = -9.0 * z_sq / 16.0
    try:
        f1f2 = special.hyp1f2(-0.5, 0.75, 1.25, arg)
    except ValueError:
        return 0.0
    return f1f2 - 1.0

# --- Worker Function for Parallel Processing (Corrected) ---
def calculate_pdf_row(la, lj_grid_vals, f_pbh, M1, M2, M_avg, sigma_M, N_y):
    """Calculates a single row (all j's for one a) of the PDF grid."""
    N_j = len(lj_grid_vals)
    pdf_row = np.zeros(N_j)
    
    # 'la' is now a SCALAR, as intended by the parallel 'delayed' call
    a = 10**la
    
    # --- FIX: Moved calculations that depend on scalar 'a' *inside* the function ---
    X = calculate_X_from_a(a, f_pbh, M_avg)
    P_a_factor = np.exp(-X) * (a**(-0.25)) if a > 0 else 0.0
    if not (np.isfinite(P_a_factor) and P_a_factor > 0): 
        return pdf_row

    j0 = calculate_j0(a, f_pbh, M1, M2, M_avg)
    sigma_jM_sq_term = (3.0/10.0) * (sigma_M**2 / f_pbh**2) * (j0**2)
    # --- END FIX ---

    # Helper for inner integral (over mass)
    def integrand_m(m, u, j_val, j0_val, N_y_val, M_avg):
        if m <= m_min_psi or m >= m_max_psi: return 0.0
        z_F = (m / M_avg) * (1.0 / N_y_val) * (j0_val / j_val) * u
        F_val = F_z_func(z_F)
        psi_val = mass_func_psi_normalized(m)
        if psi_val <= 0: return 0.0
        return (psi_val / m) * F_val

    # --- Optimization Switch ---
    Ny_threshold = 0.1 
    use_simplified_model = (N_y < Ny_threshold)
    # ---

    for k, lj in enumerate(lj_grid_vals):
        j = 10**lj
        if not (a > 0 and j > 1e-9 and j <= 1.0): continue

        P_j_given_a = 0.0
        if use_simplified_model:
            # --- FAST PATH (Eq. 2.32) ---
            jX_sq = j0**2 + sigma_jM_sq_term
            if jX_sq <= 0: continue
            jX = np.sqrt(jX_sq)
            gamma = j / jX
            denominator_term = 1.0 + gamma**2
            if denominator_term <= 0: continue
            P_gamma = (gamma**2) / (denominator_term**1.5)
            P_j_given_a = P_gamma / j
        else:
            # --- SLOW PATH (Eq. 2.31) ---
            def integrand_u(u):
                if u < 0: return 0.0
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        integral_m_val, _ = integrate.quad(
                            integrand_m, m_min_psi, m_max_psi, 
                            args=(u, j, j0, N_y, M_avg), limit=50, epsabs=1e-5, epsrel=1e-5
                        )
                except Exception: integral_m_val = 0.0
                K_m = -N_y * M_avg * integral_m_val
                K_sigma = -u**2 * (sigma_jM_sq_term / j**2)
                K_total = K_m + K_sigma
                if not np.isfinite(K_total): return 0.0
                return u * special.j0(u) * np.exp(K_total)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    P_j_integral, _ = integrate.quad(integrand_u, 0, 50.0, limit=100, epsabs=1e-5, epsrel=1e-5)
            except Exception: P_j_integral = 0.0
            P_j_given_a = (1.0 / j) * P_j_integral
            
        if not (np.isfinite(P_j_given_a) and P_j_given_a > 0): continue
            
        P_a_j_unnorm = P_j_given_a * P_a_factor
        pdf_row[k] = P_a_j_unnorm * a * j * (np.log(10)**2)
            
    return pdf_row

# --- Main Precomputation Function ---
def precompute_and_save_grid(M1, M2, f_pbh, 
                             N_a_grid, N_j_grid,
                             a_min_grid, a_max_grid,
                             j_min_grid, j_max_grid,
                             n_jobs=-1):
    
    la_grid_vals = np.linspace(np.log10(a_min_grid), np.log10(a_max_grid), N_a_grid)
    lj_grid_vals = np.linspace(np.log10(j_min_grid), np.log10(j_max_grid), N_j_grid)
    
    print(f"--- Starting Precomputation ---")
    print(f"Pair: M1={M1:.1e}, M2={M2:.1e}, f={f_pbh:.1e}")
    print(f"Grid: {N_a_grid} (a) x {N_j_grid} (j) = {N_a_grid * N_j_grid} points")
    
    n_cores_avail = multiprocessing.cpu_count()
    n_cores_to_use = n_cores_avail if n_jobs == -1 else min(n_jobs, n_cores_avail)
    print(f"Using {n_cores_to_use} CPU cores for parallel processing.")
    
    M_avg, M_avg2 = calculate_analytical_moments()
    sigma_M = sigma_eq_val
    N_y = calculate_N_y(M1, M2, M_avg, f_pbh, sigma_M)
    print(f"Params: <m>={M_avg:.3g}, sigma_M={sigma_M:.3g}, N_bar(y)={N_y:.3g}")
    
    start_time = time.time()
    
    # Parallel() maps the function calculate_pdf_row to EACH element in la_grid_vals
    results_list = Parallel(n_jobs=n_cores_to_use, verbose=10)(
        delayed(calculate_pdf_row)(la, lj_grid_vals, f_pbh, M1, M2, M_avg, sigma_M, N_y)
        for la in la_grid_vals
    )

    pdf_grid = np.array(results_list)
    end_time = time.time()
    print(f"... Precomputation finished in {end_time - start_time:.2f} seconds.")

    integral_val = np.trapz(np.trapz(pdf_grid, lj_grid_vals, axis=1), la_grid_vals, axis=0)
    
    if integral_val > 0:
        pdf_grid /= integral_val
        print(f"Grid normalized by integral value: {integral_val:.3g}")
    else:
        print("Warning: Grid integral is zero. PDF will not be normalized.")
    
    filename = f"pdf_grid_M1_{M1:.1e}_M2_{M2:.1e}_f_{f_pbh:.1e}.npz"
    np.savez_compressed(filename, 
              la_grid=la_grid_vals, 
              lj_grid=lj_grid_vals, 
              pdf_grid=pdf_grid
             )
    print(f"Grid saved to {filename}")
    return filename

# --- Main execution ---
if __name__ == "__main__":
    
    mass_pairs_to_compute = [
        (1.0, 1.0),
        (1e-4, 1e-4),
        (1.0, 1e-4)
    ]
    f_pbh_values_to_compute = np.logspace(-4., -1., 5) 
    
    N_a_grid = 150 
    N_j_grid = 150 
    
    a_min_grid = 5e-5
    a_max_grid = 0.1
    j_min_grid = 1e-6
    j_max_grid = 1.0
    
    print(f"Starting batch precomputation for {len(mass_pairs_to_compute) * len(f_pbh_values_to_compute)} total files...")
    
    for M1, M2 in mass_pairs_to_compute:
        for f_val in f_pbh_values_to_compute:
            
            filename = f"pdf_grid_M1_{M1:.1e}_M2_{M2:.1e}_f_{f_val:.1e}.npz"
            if os.path.exists(filename):
                print(f"File {filename} already exists. Skipping.")
                continue
                
            precompute_and_save_grid(M1, M2, f_val,
                                     N_a_grid, N_j_grid,
                                     a_min_grid, a_max_grid,
                                     j_min_grid, j_max_grid,
                                     n_jobs=-1)

    print("\nAll precomputation complete.")
