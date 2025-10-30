# pbh_merger_rate_callable.py
# REVISED: Precompute disrupting fraction matrix for speed.

import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.special
import os
import warnings
import time # For timing precomputation

# --- Global Constants ---
sigma_eq = np.sqrt(0.005)

# --- Function to Calculate Disrupting Fraction (Helper) ---
def calculate_disrupting_fraction(m1, m2, psi_interp_func, m_grid, avg_mass):
    """Calculates disrupting fraction F_disrupt = <m> * integral_{m3>=max(m1,m2)} (psi(m3)/m3) dm3."""
    m_threshold = max(m1, m2)
    integrand_disrupt = lambda m: psi_interp_func(m) / m if m > 0 else 0.0
    m_grid_disrupt = m_grid[m_grid >= m_threshold]
    if len(m_grid_disrupt) < 2:
        integral_disrupt = 0.0
    else:
        psi_vals_disrupt = np.array([integrand_disrupt(m) for m in m_grid_disrupt])
        # Use np.trapz for direct numpy array integration
        integral_disrupt = np.trapz(psi_vals_disrupt, m_grid_disrupt)
        integral_disrupt = max(0.0, integral_disrupt)
    if not (np.isfinite(avg_mass) and avg_mass > 0): return 1.0 # Fallback
    fraction = avg_mass * integral_disrupt
    return max(0.0, min(fraction, 1.0))

# --- Helper Functions for Suppression Factors (NNN_effective modified, S_E takes disrupt_frac) ---
def NNN_effective(mi, mj, avM, fpbh0_tot, sigma_eq_val, disrupting_fraction):
    """Calculates the EFFECTIVE N(y) using the disrupting fraction."""
    if not (np.isfinite(avM) and avM > 0): return 0.0
    denom = fpbh0_tot + sigma_eq_val
    if not (np.isfinite(denom) and denom > 0): return 0.0
    N_orig = (mi + mj) / avM * fpbh0_tot / denom
    if not np.isfinite(N_orig): N_orig = 0.0
    N_eff = N_orig * disrupting_fraction # Scale by precomputed fraction
    return N_eff

# S_L remains the same
def S_L(fpbh0_tot):
    ht = np.maximum(fpbh0_tot, 1e-99);
    try:
        log_ht = np.log(ht); log_ht_sq = log_ht**2; exp_term = np.exp(0.03 * log_ht_sq)
        t1_term = np.power(ht, -0.65); t1 = t1_term * exp_term
        if not np.isfinite(t1): t1 = np.inf
        xt = np.min([1.0, 0.01*t1])
    except (ValueError, OverflowError, RuntimeWarning): xt = 0.0
    return xt if np.isfinite(xt) else 0.0

# CCC remains the same
def CCC(fpbh0_tot, avM, avM2, sigma_eq_val):
    # ... (Keep previous corrected CCC implementation) ...
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


# S_E now takes disrupting_fraction as argument
def S_E(mi, mj, fpbh0_tot, avM, avM2, sigma_eq_val, disrupting_fraction):
    """Suppression factor S_E using precomputed disrupting_fraction."""
    if not (np.isfinite(avM) and avM > 0 and np.isfinite(avM2) and avM2 >= 0 and
            np.isfinite(fpbh0_tot) and fpbh0_tot > 0 and np.isfinite(sigma_eq_val) and sigma_eq_val > 0): return 0.0

    # Use effective NNN calculated with the provided fraction
    nnn_eff_val = NNN_effective(mi, mj, avM, fpbh0_tot, sigma_eq_val, disrupting_fraction)
    ccc_val = CCC(fpbh0_tot, avM, avM2, sigma_eq_val)
    if not np.isfinite(ccc_val): ccc_val = 1e30 # Cap

    avM_sq = avM**2; avM2_over_avM_sq = avM2 / avM_sq
    t1_factor = (np.sqrt(np.pi) * (5.0 / 6.0)**(21.0 / 74.0) / scipy.special.gamma(29.0 / 37.0))
    denom_t2 = (nnn_eff_val + ccc_val) # Use effective NNN here
    if denom_t2 <= 0: return 0.0
    fpbh_sq = fpbh0_tot**2; sigma_sq = sigma_eq_val**2
    term_in_t2 = (avM2_over_avM_sq / denom_t2 + sigma_sq / fpbh_sq)
    if term_in_t2 <= 0: return 0.0
    try: t2_factor = np.power(term_in_t2, -21.0 / 74.0)
    except (ValueError, OverflowError): return 0.0
    t3_factor = np.exp(-nnn_eff_val) # Use effective NNN in exponent

    result = t1_factor * t2_factor * t3_factor
    return result if np.isfinite(result) else 0.0


# --- Rate Density Function (now takes disrupting_fraction) ---
def rate_density(mi, mj, f_total, psi_interp, avM, avM2, sigma_eq_val, disrupting_fraction):
    """Calculates dR / (dln m1 dln m2)"""
    if mi <= 0 or mj <= 0 or f_total <= 0: return 0.0
    const_factor = 1.6e6; f_factor = np.power(f_total, 53.0 / 37.0) if f_total > 0 else 0.0
    M_tot = mi + mj; mass_factor = np.power(M_tot, -32.0 / 37.0) if M_tot > 0 else 0.0
    eta = (mi * mj) / (M_tot**2) if M_tot > 0 else 0.0
    eta_factor = np.power(eta, -34.0 / 37.0) if eta > 1e-99 else 0.0
    sl_val = S_L(f_total)
    # Pass precomputed disrupting_fraction to S_E
    se_val = S_E(mi, mj, f_total, avM, avM2, sigma_eq_val, disrupting_fraction)
    suppression_factor = sl_val * se_val
    try:
        psi_mi = psi_interp(mi); psi_mj = psi_interp(mj)
        psi_mi = max(0.0, psi_mi if np.isfinite(psi_mi) else 0.0)
        psi_mj = max(0.0, psi_mj if np.isfinite(psi_mj) else 0.0)
    except ValueError: psi_mi = 0.0; psi_mj = 0.0
    psi_factor = psi_mi * psi_mj
    rate_dm = const_factor * f_factor * eta_factor * mass_factor * suppression_factor * psi_factor
    rate_dln = rate_dm * mi * mj
    rate_floor = 1e-99
    final_rate = rate_dln if np.isfinite(rate_dln) else 0.0
    return max(final_rate, rate_floor)


# --- Main Calculation Function (PRECOMPUTES disrupt_fraction_matrix) ---
def calculate_merger_rate_matrix(f_total, psi_interp_func, mlist, # Use interpolator directly
                                  m1_peak=None, m2_peak=None, # Peaks optional
                                  use_analytical_moments=False): # Flag
    """
    Calculates the PBH merger rate matrix dR/(dln m1 dln m2).
    Includes mass-ratio dependent disruption suppression (precomputed).
    """
    global sigma_eq
    print(f"Calculating analytical rate matrix for f_PBH = {f_total:.3e}...")
    mlist = np.sort(np.asarray(mlist)); valid_m_mask = mlist > 0; mlist_valid = mlist[valid_m_mask]
    N = len(mlist) # Original length of mlist
    if len(mlist_valid) < 2: return mlist, np.zeros((N, N)), f_total, 0.0

    # --- Calculate Moments ---
    # ... (Keep previous corrected moment calculation using analytical or trapz) ...
    avM = np.nan; avM2 = np.nan
    if use_analytical_moments and m1_peak is not None and m2_peak is not None:
        print(f"  Using analytical moments for bimodal peaks ({m1_peak:.1e}, {m2_peak:.1e}).")
        if m1_peak <= 0 or m2_peak <= 0: raise ValueError("Peak masses must be positive.")
        denom_avM = 0.5/m1_peak + 0.5/m2_peak
        if denom_avM <= 0: raise ValueError("<m> calculation failed.")
        avM = 1.0 / denom_avM
        avM2 = 0.5 * m1_peak**2 + 0.5 * m2_peak**2
    else:
        print("  Calculating moments numerically using trapz...")
        psi_on_mlist = psi_interp_func(mlist)
        psi_on_mlist[psi_on_mlist < 0] = 0.0
        integrand_inv_m = np.zeros_like(mlist);
        integrand_inv_m[valid_m_mask] = psi_on_mlist[valid_m_mask] / mlist[valid_m_mask]
        integral_inv_m = scipy.integrate.trapz(integrand_inv_m[valid_m_mask], mlist_valid)
        if integral_inv_m <= 1e-99: raise ValueError("<m> calc failed (integral <= 0).")
        avM = 1.0 / integral_inv_m
        integrand_m2 = mlist**2 * psi_on_mlist
        avM2 = scipy.integrate.trapz(integrand_m2[valid_m_mask], mlist_valid)
        if not np.isfinite(avM2) or avM2 < 0: raise ValueError("<m^2> calc failed.")
    print(f"    <m>   = {avM:.3g}"); print(f"    <m^2> = {avM2:.3g}")
    if avM > 0: print(f"    <m^2>/<m>^2 = {avM2/avM**2:.3g}")

    fpbh0_for_suppression = f_total; sigma_eq_val = sigma_eq

    # --- PRECOMPUTE Disrupting Fraction Matrix ---
    print("    Precomputing disrupting fraction matrix...")
    start_time_precomp = time.time()
    disrupt_fraction_matrix = np.ones((N, N)) # Initialize to 1 (no suppression)
    for i in range(N):
        for j in range(N):
            mi = mlist[i]; mj = mlist[j]
            # Calculate only if masses are valid
            if mi > 0 and mj > 0:
                disrupt_fraction_matrix[i, j] = calculate_disrupting_fraction(
                    mi, mj, psi_interp_func, mlist, avM # Pass full mlist for integration range
                )
    end_time_precomp = time.time()
    print(f"    ... Disrupting fraction precomputation done ({end_time_precomp - start_time_precomp:.2f} s).")
    # --- END PRECOMPUTATION ---

    # --- Calculate Rate Matrix ---
    print("    Calculating rate matrix elements...")
    start_time_matrix = time.time()
    rate_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mi = mlist[i]; mj = mlist[j]
            # Retrieve precomputed fraction
            disrupt_frac = disrupt_fraction_matrix[i, j]
            rate_matrix[i, j] = rate_density(mi, mj, f_total, psi_interp_func, avM, avM2, sigma_eq_val, disrupt_frac)
    end_time_matrix = time.time()
    print(f"    ... Rate matrix calculation done ({end_time_matrix - start_time_matrix:.2f} s).")

    # --- Calculate Total Integrated Rate ---
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
    else: print("Warning: <2 valid mass points for total rate integration.")

    if not np.isfinite(total_rate) or total_rate < 0: total_rate = 0.0
    elif total_rate > 1e20: print(f"Warning: Total rate very large ({total_rate:.3g}).")

    print(f"... Analytical calculation done. Total Rate = {total_rate:.3g} Gpc^-3 yr^-1")
    fpbh0_to_return = f_total
    return mlist, rate_matrix, fpbh0_to_return, total_rate

# --- Main execution block ---
if __name__ == '__main__':
    # ... (Example usage remains the same) ...
    pass
