# pbh_merger_rate_callable.py
# REVISED: Reverted disrupting fraction calculation back to trapz for speed,
# keeping precomputation structure.

import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.special
import os
import warnings
import time

# --- Global Constants ---
sigma_eq = np.sqrt(0.005)

# --- Function to Calculate Disrupting Fraction (Helper - USES TRAPZ AGAIN) ---
def calculate_disrupting_fraction(m1, m2, psi_interp_func, m_grid, avg_mass):
    """
    Calculates disrupting fraction F_disrupt = <m> * integral_{m3>=max(m1,m2)} (psi(m3)/m3) dm3
    using np.trapz over the provided grid.
    """
    m_threshold = max(m1, m2)

    # Find the portion of the grid >= threshold
    mask_disrupt = m_grid >= m_threshold
    m_grid_disrupt = m_grid[mask_disrupt]

    integral_disrupt = 0.0
    if len(m_grid_disrupt) >= 2: # Need at least 2 points for trapz
        # Evaluate psi/m only on the relevant subgrid
        psi_vals_grid = psi_interp_func(m_grid_disrupt)
        psi_vals_grid[psi_vals_grid < 0] = 0.0 # Ensure non-negative
        
        integrand_vals = np.zeros_like(m_grid_disrupt)
        valid_m_mask = m_grid_disrupt > 0
        integrand_vals[valid_m_mask] = psi_vals_grid[valid_m_mask] / m_grid_disrupt[valid_m_mask]

        # Use trapz for numerical integration over the subgrid
        integral_disrupt = np.trapz(integrand_vals, m_grid_disrupt)
        integral_disrupt = max(0.0, integral_disrupt) # Ensure non-negative
    # else: If less than 2 points >= threshold, integral is effectively 0

    # F_disrupt = <m> * integral
    if not (np.isfinite(avg_mass) and avg_mass > 0): return 1.0 # Fallback
    fraction = avg_mass * integral_disrupt
    return max(0.0, min(fraction, 1.0)) # Ensure 0 <= fraction <= 1


# --- Helper Functions for Suppression Factors ---
# ... (Keep NNN_effective, S_L, CCC, S_E definitions from previous version) ...
def NNN_effective(mi, mj, avM, fpbh0_tot, sigma_eq_val, disrupting_fraction):
    if not (np.isfinite(avM) and avM > 0): return 0.0
    denom = fpbh0_tot + sigma_eq_val
    if not (np.isfinite(denom) and denom > 0): return 0.0
    N_orig = (mi + mj) / avM * fpbh0_tot / denom
    if not np.isfinite(N_orig): N_orig = 0.0
    N_eff = N_orig * disrupting_fraction
    return N_eff

def S_L(fpbh0_tot):
    ht = np.maximum(fpbh0_tot, 1e-99);
    try:
        log_ht = np.log(ht); log_ht_sq = log_ht**2; exp_term = np.exp(0.03 * log_ht_sq)
        t1_term = np.power(ht, -0.65); t1 = t1_term * exp_term
        if not np.isfinite(t1): t1 = np.inf
        xt = np.min([1.0, 0.01*t1])
    except (ValueError, OverflowError, RuntimeWarning): xt = 0.0
    return xt if np.isfinite(xt) else 0.0

def CCC(fpbh0_tot, avM, avM2, sigma_eq_val):
    if not (np.isfinite(avM) and avM > 0 and np.isfinite(avM2) and avM2 >= 0 and
            np.isfinite(fpbh0_tot) and fpbh0_tot > 0 and np.isfinite(sigma_eq_val) and sigma_eq_val > 0): return 0.0
    avM_sq = avM**2; avM2_over_avM_sq = avM2 / avM_sq
    sigma_sq = sigma_eq_val**2; fpbh_sq = fpbh0_tot**2
    t1 = (fpbh_sq * avM2_over_avM_sq / sigma_sq)
    ht0 = scipy.special.gamma(29.0/37.0) / np.sqrt(np.pi)
    arg_hyperu = 5.0 * fpbh_sq / (6.0 * sigma_sq)
    ht1 = 0.0
    try: # SciPy hyperu
        ht1 = scipy.special.hyperu(21.0/74.0, 1.0/2.0, arg_hyperu)
        if not np.isfinite(ht1): raise ValueError("scipy hyperu not finite")
    except (ValueError, OverflowError, TypeError):
         if arg_hyperu > 100: ht1 = np.power(arg_hyperu, -21.0/74.0) # Asymptotic
         else: # Try mpmath
              try: import mpmath; mpmath.mp.dps = 25; ht1 = float(mpmath.hyperu(21.0/74.0, 1.0/2.0, arg_hyperu));
              except Exception: ht1 = 0.0
    term_in_brackets = ht0 * ht1
    if term_in_brackets <= 1e-99: return np.inf
    try: power_val = np.power(term_in_brackets, -74.0 / 21.0);
    except Exception: return np.inf
    denominator = power_val - 1.0
    if abs(denominator) < 1e-99: return np.inf
    ccc_val = t1 / denominator
    return ccc_val if np.isfinite(ccc_val) else 0.0

def S_E(mi, mj, fpbh0_tot, avM, avM2, sigma_eq_val, disrupting_fraction):
    """Suppression factor S_E using precomputed disrupting_fraction."""
    if not (np.isfinite(avM) and avM > 0 and np.isfinite(avM2) and avM2 >= 0 and
            np.isfinite(fpbh0_tot) and fpbh0_tot > 0 and np.isfinite(sigma_eq_val) and sigma_eq_val > 0): return 0.0
    nnn_eff_val = NNN_effective(mi, mj, avM, fpbh0_tot, sigma_eq_val, disrupting_fraction)
    ccc_val = CCC(fpbh0_tot, avM, avM2, sigma_eq_val)
    if not np.isfinite(ccc_val): ccc_val = 1e30 # Cap
    avM_sq = avM**2; avM2_over_avM_sq = avM2 / avM_sq
    t1_factor = (np.sqrt(np.pi) * (5.0 / 6.0)**(21.0 / 74.0) / scipy.special.gamma(29.0 / 37.0))
    denom_t2 = (nnn_eff_val + ccc_val)
    if denom_t2 <= 0: return 0.0
    fpbh_sq = fpbh0_tot**2; sigma_sq = sigma_eq_val**2
    term_in_t2 = (avM2_over_avM_sq / denom_t2 + sigma_sq / fpbh_sq)
    if term_in_t2 <= 0: return 0.0
    try: t2_factor = np.power(term_in_t2, -21.0 / 74.0)
    except (ValueError, OverflowError): return 0.0
    exp_arg = -nnn_eff_val; t3_factor = np.exp(exp_arg) if exp_arg > -700 else 0.0
    result = t1_factor * t2_factor * t3_factor
    return result if np.isfinite(result) else 0.0

# --- Rate Density Function ---
def rate_density(mi, mj, f_total, psi_interp, avM, avM2, sigma_eq_val, disrupting_fraction):
    """Calculates dR / (dln m1 dln m2)"""
    # ... (Keep the previous corrected rate_density implementation) ...
    if mi <= 0 or mj <= 0 or f_total <= 0: return 0.0
    const_factor = 1.6e6; f_factor = np.power(f_total, 53.0 / 37.0) if f_total > 0 else 0.0
    M_tot = mi + mj; mass_factor = np.power(M_tot, -32.0 / 37.0) if M_tot > 0 else 0.0
    eta = (mi * mj) / (M_tot**2) if M_tot > 0 else 0.0
    eta_factor = np.power(eta, -34.0 / 37.0) if eta > 1e-99 else 0.0
    sl_val = S_L(f_total)
    se_val = S_E(mi, mj, f_total, avM, avM2, sigma_eq_val, disrupting_fraction) # Pass disrupt_frac
    suppression_factor = sl_val * se_val
    try: psi_mi = psi_interp(mi); psi_mj = psi_interp(mj)
    except ValueError: psi_mi = 0.0; psi_mj = 0.0
    psi_mi = max(0.0, psi_mi if np.isfinite(psi_mi) else 0.0)
    psi_mj = max(0.0, psi_mj if np.isfinite(psi_mj) else 0.0)
    psi_factor = psi_mi * psi_mj
    rate_dm = const_factor * f_factor * eta_factor * mass_factor * suppression_factor * psi_factor
    rate_dln = rate_dm * mi * mj
    rate_floor = 1e-99
    final_rate = rate_dln if np.isfinite(rate_dln) else 0.0
    return max(final_rate, rate_floor)

# --- Main Calculation Function ---
def calculate_merger_rate_matrix(f_total, psi_interp_func, mlist, # Original args
                                  avM, avM2, # Precomputed moments
                                  disrupt_fraction_matrix): # Precomputed matrix
    """
    Calculates the PBH merger rate matrix dR/(dln m1 dln m2).
    Uses PRECOMPUTED moments and disrupting fraction matrix.
    """
    global sigma_eq
    # print(f"Calculating analytical rate matrix for f_PBH = {f_total:.3e}...") # Reduce print

    mlist = np.sort(np.asarray(mlist)); valid_m_mask = mlist > 0; mlist_valid = mlist[valid_m_mask]
    N = len(mlist)
    if len(mlist_valid) < 2: return mlist, np.zeros((N, N)), f_total, 0.0

    fpbh0_for_suppression = f_total; sigma_eq_val = sigma_eq

    # --- Calculate Rate Matrix using precomputed disrupt_fraction_matrix ---
    # print("    Calculating rate matrix elements...") # Reduce print
    start_time_matrix = time.time()
    rate_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mi = mlist[i]; mj = mlist[j]
            disrupt_frac = disrupt_fraction_matrix[i, j]
            rate_matrix[i, j] = rate_density(mi, mj, f_total, psi_interp_func, avM, avM2, sigma_eq_val, disrupt_frac)
    end_time_matrix = time.time()
    # print(f"    ... Rate matrix calculation done ({end_time_matrix - start_time_matrix:.2f} s).") # Reduce print

    # --- Calculate Total Integrated Rate (using trapz over grid) ---
    total_rate = 0.0
    if len(mlist_valid) >= 2:
        log_m = np.log(mlist_valid)
        rate_matrix_valid = rate_matrix[valid_m_mask][:, valid_m_mask]
        rate_matrix_finite = np.nan_to_num(rate_matrix_valid)
        try:
            integral_over_j = np.trapz(rate_matrix_finite, x=log_m, axis=1)
            total_rate = np.trapz(integral_over_j, x=log_m, axis=0)
            total_rate = max(0.0, total_rate)
        except Exception as e:
            print(f"Warning: 2D trapz integration failed: {e}"); total_rate = np.nan
    # else: print("Warning: <2 valid mass points for total rate integration.") # Reduce print

    if not np.isfinite(total_rate) or total_rate < 0: total_rate = 0.0
    # elif total_rate > 1e20: print(f"Warning: Total rate very large ({total_rate:.3g}).") # Reduce print

    # print(f"... Analytical calculation done. Total Rate = {total_rate:.3g} Gpc^-3 yr^-1") # Reduce print
    fpbh0_to_return = f_total
    return mlist, rate_matrix, fpbh0_to_return, total_rate

# --- Main execution block for example/testing ---
if __name__ == '__main__':
    import scipy.stats
    print("\nRunning pbh_merger_rate_callable.py example (with precomputation)...")

    # Define psi, grid, interpolator
    m1_peak_ex = 1.0; m2_peak_ex = 1.0e-4; sigma_rel_ex = 0.01
    sigma_g1_ex = max(sigma_rel_ex * m1_peak_ex, 1e-12 * m1_peak_ex)
    sigma_g2_ex = max(sigma_rel_ex * m2_peak_ex, 1e-12 * m2_peak_ex)
    def mass_func_psi_norm_ex(m):
        pdf1 = 0.5 * scipy.stats.norm.pdf(m, loc=m1_peak_ex, scale=sigma_g1_ex)
        pdf2 = 0.5 * scipy.stats.norm.pdf(m, loc=m2_peak_ex, scale=sigma_g2_ex)
        result = pdf1 + pdf2
        return np.maximum(0.0, result) if isinstance(m, np.ndarray) else max(0.0, result)
    m_min_psi_ex = m2_peak_ex / 1000; m_max_psi_ex = m1_peak_ex * 10
    num_points_psi_ex = 1000
    mass_grid_psi_ex = np.logspace(np.log10(m_min_psi_ex), np.log10(m_max_psi_ex), num_points_psi_ex)
    psi_vals_grid_ex = mass_func_psi_norm_ex(mass_grid_psi_ex)
    norm_check_ex = scipy.integrate.trapz(psi_vals_grid_ex, mass_grid_psi_ex)
    print(f"Example psi normalization check: {norm_check_ex:.4g}")
    unique_m_ex, unique_idx_ex = np.unique(mass_grid_psi_ex, return_index=True)
    if len(unique_m_ex) < 2: raise ValueError("Example grid issue?")
    psi_interp_ex = scipy.interpolate.interp1d(unique_m_ex, psi_vals_grid_ex[unique_idx_ex], bounds_error=False, fill_value=0.0)

    # Precompute Moments
    use_analytical_ex = True
    if use_analytical_ex:
        denom_avM_ex = 0.5/m1_peak_ex + 0.5/m2_peak_ex; avM_ex = 1.0 / denom_avM_ex
        avM2_ex = 0.5 * m1_peak_ex**2 + 0.5 * m2_peak_ex**2
    else: # Numerical moments using quad
        integrand_inv_m_ex = lambda m: mass_func_psi_norm_ex(m) / m if m > 0 else 0.0
        integral_inv_m_ex, _ = scipy.integrate.quad(integrand_inv_m_ex, m_min_psi_ex, m_max_psi_ex, limit=200)
        avM_ex = 1.0 / integral_inv_m_ex
        integrand_m2_ex = lambda m: m**2 * mass_func_psi_norm_ex(m) if m > 0 else 0.0
        avM2_ex, _ = scipy.integrate.quad(integrand_m2_ex, m_min_psi_ex, m_max_psi_ex, limit=200)
    print(f"Example moments: <m> = {avM_ex:.3g}, <m^2> = {avM2_ex:.3g}")

    # Precompute Disrupt Fraction Matrix using trapz (faster)
    test_mlist = np.logspace(-5, 2, 150) # Use the grid size intended for calculation
    N_test = len(test_mlist)
    disrupt_frac_matrix_ex = np.ones((N_test, N_test))
    print("Precomputing disrupt fraction matrix for example (using trapz)...")
    for i in range(N_test):
        for j in range(N_test):
            mi = test_mlist[i]; mj = test_mlist[j]
            if mi > 0 and mj > 0:
                # Pass the grid used for psi definition for integration range
                disrupt_frac_matrix_ex[i, j] = calculate_disrupting_fraction(
                    mi, mj, psi_interp_ex, mass_grid_psi_ex, avM_ex
                )
    print("... Precomputation done.")

    # Run calculation for one f value
    test_f_total = 0.001
    mass_out, rate_mat_out, fpbh0_out, total_rate_out = calculate_merger_rate_matrix(
        f_total=test_f_total,
        psi_interp_func=psi_interp_ex,
        mlist=test_mlist, # Use the same grid for rate matrix calculation
        m_min_psi=m_min_psi_ex, m_max_psi=m_max_psi_ex, # Pass grid limits anyway (not used if moments precomp)
        avM=avM_ex, avM2=avM2_ex,
        disrupt_fraction_matrix=disrupt_frac_matrix_ex
    )

    print(f"\nExample results for f={test_f_total}: Total Rate = {total_rate_out:.4g} Gpc^-3 yr^-1")
    idx1_ex = np.argmin(np.abs(test_mlist - m1_peak_ex)); idx2_ex = np.argmin(np.abs(test_mlist - m2_peak_ex))
    print(f"Rate Density M1-M1: {rate_mat_out[idx1_ex, idx1_ex]:.3g}")
    print(f"Rate Density M2-M2: {rate_mat_out[idx2_ex, idx2_ex]:.3g}")
    print(f"Rate Density M1-M2: {rate_mat_out[idx1_ex, idx2_ex]:.3g}")

    output_dir_example = "."; output_file_example = os.path.join(output_dir_example, f"pbh_merger_rate_example_f{test_f_total:.1e}.npz")
    np.savez(output_file_example, m=mass_out, rate=rate_mat_out, fpbh=test_f_total, total_rate=total_rate_out)
    print(f"Example PBH merger rate saved to {output_file_example}")
