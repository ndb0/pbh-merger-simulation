# pbh_merger_rate_callable.py
# Calculates analytical PBH merger rate dR/(dln m1 dln m2) and total rate.
# Uses analytical moments for sharp bimodal Gaussian psi(m).

import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.special
import os

# --- Global Constants ---
sigma_eq = np.sqrt(0.005) # Raidal paper value, sqrt based on analytical script

# --- Helper Functions for Suppression Factors ---
def NNN(mi, mj, avM, fpbh0_tot, sigma_eq_val):
    """Calculates N(y) approximation component."""
    # Ensure avM and denominator are positive and finite
    if not (np.isfinite(avM) and avM > 0): return 0.0
    denom = fpbh0_tot + sigma_eq_val
    if not (np.isfinite(denom) and denom > 0): return 0.0
    val = (mi + mj) / avM * fpbh0_tot / denom
    return val if np.isfinite(val) else 0.0

def S_L(fpbh0_tot):
    """Suppression factor S_L based on Eq. 60 from 2404.08416v2."""
    ht = np.maximum(fpbh0_tot, 1e-99) # Ensure positive argument for log
    try:
        log_ht = np.log(ht)
        log_ht_sq = log_ht**2
        exp_term = np.exp(0.03 * log_ht_sq)
        t1_term = np.power(ht, -0.65)
        t1 = t1_term * exp_term
        # Check for overflow/NaN before min
        if not np.isfinite(t1): t1 = np.inf
        xt = np.min([1.0, 0.01*t1])
    except (ValueError, OverflowError):
        xt = 0.0 # Return 0 if calculation fails
    return xt if np.isfinite(xt) else 0.0

def CCC(fpbh0_tot, avM, avM2, sigma_eq_val):
    """Calculates C factor related to S_E based on Eq. 47 from 2404.08416v2."""
    # Ensure inputs are valid
    if not (np.isfinite(avM) and avM > 0 and
            np.isfinite(avM2) and avM2 >= 0 and
            np.isfinite(fpbh0_tot) and fpbh0_tot > 0 and
            np.isfinite(sigma_eq_val) and sigma_eq_val > 0):
        return 0.0

    avM2_over_avM_sq = avM2 / avM**2
    sigma_sq = sigma_eq_val**2
    fpbh_sq = fpbh0_tot**2

    # Term 1 in C calculation
    t1_num = fpbh_sq * avM2_over_avM_sq
    t1_den = sigma_sq
    if t1_den <= 0: return 0.0
    t1 = t1_num / t1_den

    # Term involving HypergeometricU
    ht0 = scipy.special.gamma(29.0/37.0) / np.sqrt(np.pi)
    arg_hyperu_num = 5.0 * fpbh_sq
    arg_hyperu_den = 6.0 * sigma_sq
    if arg_hyperu_den <= 0: return 0.0
    arg_hyperu = arg_hyperu_num / arg_hyperu_den

    try:
        ht1 = scipy.special.hyperu(21.0/74.0, 1.0/2.0, arg_hyperu)
        if not np.isfinite(ht1): raise ValueError("hyperu not finite")
    except (ValueError, OverflowError, TypeError):
        # Use asymptotic approx U(a,b,z) ~ z^-a for large z
        ht1 = np.power(arg_hyperu, -21.0/74.0) if arg_hyperu > 100 else 0.0

    term_in_brackets = ht0 * ht1
    if term_in_brackets <= 0: return 0.0 # Avoid issues with power

    try:
        # Power exponent calculation
        power_base = term_in_brackets
        power_exp = -74.0 / 21.0
        power_val = np.power(power_base, power_exp)
        if not np.isfinite(power_val): raise OverflowError
    except (ValueError, OverflowError):
        # If base is tiny, result is huge; if base large, result tiny.
        # Set a large cap or small floor if needed, or return indication of issue.
        return np.inf # Indicate potential issue if power blows up

    denominator = power_val - 1.0
    if abs(denominator) < 1e-99: # Avoid division by zero
        return np.inf # Or handle differently if C=0 is expected

    ccc_val = t1 / denominator
    return ccc_val if np.isfinite(ccc_val) else 0.0 # Return 0 if result is NaN/Inf


def S_E(mi, mj, fpbh0_tot, avM, avM2, sigma_eq_val):
    """Suppression factor S_E based on Eq. 46 from 2404.08416v2."""
    # Ensure inputs are valid
    if not (np.isfinite(avM) and avM > 0 and
            np.isfinite(avM2) and avM2 >= 0 and
            np.isfinite(fpbh0_tot) and fpbh0_tot > 0 and
            np.isfinite(sigma_eq_val) and sigma_eq_val > 0):
        return 0.0

    nnn_val = NNN(mi, mj, avM, fpbh0_tot, sigma_eq_val)
    ccc_val = CCC(fpbh0_tot, avM, avM2, sigma_eq_val)
    # Check if CCC returned Inf
    if not np.isfinite(ccc_val): ccc_val = 1e30 # Use a large number as proxy

    avM2_over_avM_sq = avM2 / avM**2

    t1_factor = (np.sqrt(np.pi) * (5.0 / 6.0)**(21.0 / 74.0) / scipy.special.gamma(29.0 / 37.0))

    denom_t2 = (nnn_val + ccc_val)
    if denom_t2 <= 0: return 0.0 # Denominator must be positive

    fpbh_sq = fpbh0_tot**2
    sigma_sq = sigma_eq_val**2
    term_in_t2_num = sigma_sq
    term_in_t2_den = fpbh_sq
    if term_in_t2_den <= 0: return 0.0

    term_in_t2 = (avM2_over_avM_sq / denom_t2 + term_in_t2_num / term_in_t2_den)
    if term_in_t2 <= 0: return 0.0 # Base for power must be positive

    try:
        t2_factor = np.power(term_in_t2, -21.0 / 74.0)
        if not np.isfinite(t2_factor): raise OverflowError
    except (ValueError, OverflowError):
        return 0.0 # Calculation failed

    t3_factor = np.exp(-nnn_val)

    result = t1_factor * t2_factor * t3_factor
    return result if np.isfinite(result) else 0.0


# --- Rate Density Function ---
def rate_density(mi, mj, f_total, psi_interp, avM, avM2, sigma_eq_val):
    """Calculates dR / (dln m1 dln m2) based on Eq 101 / Eq 2.35."""
    if mi <= 0 or mj <= 0 or f_total <= 0: return 0.0

    const_factor = 1.6e6 # Units: Gpc^-3 yr^-1 M_sun^2 (from Eq. 2.35 dimension)

    f_factor = np.power(f_total, 53.0 / 37.0) if f_total > 0 else 0.0

    M_tot = mi + mj
    if M_tot <= 0 : return 0.0
    mass_factor = np.power(M_tot, -32.0 / 37.0) # Assumes M in M_sun

    eta = (mi * mj) / (M_tot**2)
    eta_factor = np.power(eta, -34.0 / 37.0) if eta > 0 else 0.0

    sl_val = S_L(f_total)
    se_val = S_E(mi, mj, f_total, avM, avM2, sigma_eq_val)
    suppression_factor = sl_val * se_val

    # Mass function factor psi(m1) * psi(m2) (Units: M_sun^-2)
    try:
        psi_mi = psi_interp(mi); psi_mj = psi_interp(mj)
        psi_mi = max(0.0, psi_mi if np.isfinite(psi_mi) else 0.0)
        psi_mj = max(0.0, psi_mj if np.isfinite(psi_mj) else 0.0)
    except ValueError: psi_mi = 0.0; psi_mj = 0.0
    psi_factor = psi_mi * psi_mj

    # Combine: R_dm = const * f * eta * M * S * psi (Units: Gpc^-3 yr^-1 M_sun^-2)
    rate_dm = const_factor * f_factor * eta_factor * mass_factor * suppression_factor * psi_factor

    # Convert to dR / dln m1 dln m2 = R_dm * m1 * m2 (Units: Gpc^-3 yr^-1)
    rate_dln = rate_dm * mi * mj

    return rate_dln if np.isfinite(rate_dln) else 0.0


# --- Main Calculation Function ---
def calculate_merger_rate_matrix(f_total, psi_interp_func, mlist, # Use interpolator directly
                                  m1_peak=None, m2_peak=None, # Peaks optional for analytical moments
                                  use_analytical_moments=False): # Flag to use analytical moments
    """
    Calculates the PBH merger rate matrix dR/(dln m1 dln m2).
    Can use analytical approximations for <m> and <m^2> for sharp bimodal Gaussian
    if use_analytical_moments=True and peaks are provided. Otherwise calculates numerically.
    psi_interp_func should be the interpolator for the normalized psi(m) [int psi dm = 1].
    """
    global sigma_eq

    print(f"Calculating analytical rate matrix for f_PBH = {f_total:.3e}...")

    mlist = np.sort(np.asarray(mlist)); valid_m = mlist[mlist > 0]
    if len(valid_m) < 2: return mlist, np.zeros((len(mlist), len(mlist))), f_total, 0.0

    # --- Calculate Moments ---
    if use_analytical_moments and m1_peak is not None and m2_peak is not None:
        print(f"  Using analytical moments for bimodal peaks ({m1_peak:.1e}, {m2_peak:.1e}).")
        if m1_peak <= 0 or m2_peak <= 0: raise ValueError("Peak masses must be positive.")
        # Assuming each peak contributes 0.5 probability mass
        denom_avM = 0.5/m1_peak + 0.5/m2_peak
        if denom_avM <= 0: raise ValueError("<m> denominator calculation failed.")
        avM = 1.0 / denom_avM
        avM2 = 0.5 * m1_peak**2 + 0.5 * m2_peak**2
    else:
        print("  Calculating moments numerically using trapz...")
        # Evaluate psi on the grid from the interpolator
        psi_on_mlist = psi_interp_func(mlist)
        psi_on_mlist[psi_on_mlist < 0] = 0.0 # Ensure non-negative

        # <m> = 1 / integral(psi/m dm)
        integrand_inv_m = np.zeros_like(mlist); valid_mask = mlist > 0
        integrand_inv_m[valid_mask] = psi_on_mlist[valid_mask] / mlist[valid_mask]
        integral_inv_m = scipy.integrate.trapz(integrand_inv_m, mlist)
        if integral_inv_m <= 1e-99: raise ValueError("<m> calculation failed (integral <= 0).")
        avM = 1.0 / integral_inv_m

        # <m^2> = integral(m^2 * psi dm)
        integrand_m2 = mlist**2 * psi_on_mlist
        avM2 = scipy.integrate.trapz(integrand_m2, mlist) # This is <m^2>
        if not np.isfinite(avM2) or avM2 < 0: raise ValueError("<m^2> calculation failed.")

    print(f"    <m>   = {avM:.3g}")
    print(f"    <m^2> = {avM2:.3g}")
    # Check ratio for potential issues in S_E
    avM_sq = avM**2
    if avM_sq > 0: print(f"    <m^2>/<m>^2 = {avM2/avM_sq:.3g}")

    fpbh0_for_suppression = f_total; sigma_eq_val = sigma_eq

    # --- Calculate Rate Matrix ---
    rate_matrix = np.zeros((len(mlist), len(mlist)))
    for i in range(len(mlist)):
        for j in range(len(mlist)):
            mi = mlist[i]; mj = mlist[j]
            rate_matrix[i, j] = rate_density(mi, mj, f_total, psi_interp_func, avM, avM2, sigma_eq_val)

    # --- Calculate Total Integrated Rate ---
    total_rate = 0.0
    if len(mlist) > 1:
        log_m = np.log(mlist)
        delta_log_m_mid = 0.5 * (np.diff(log_m, prepend=log_m[0]-(log_m[1]-log_m[0])) + np.diff(log_m, append=log_m[-1]+(log_m[-1]-log_m[-2])))
        dlogm_i, dlogm_j = np.meshgrid(delta_log_m_mid, delta_log_m_mid, indexing='ij')
        rate_matrix_finite = np.nan_to_num(rate_matrix) # Replace NaN/Inf with 0
        total_rate = np.sum(rate_matrix_finite * dlogm_i * dlogm_j)

    print(f"... Analytical calculation done. Total Rate = {total_rate:.3g} Gpc^-3 yr^-1")
    return mlist, rate_matrix, fpbh0_for_suppression, total_rate

# --- Main execution block for example/testing ---
if __name__ == '__main__':
    import scipy.stats # Needed for example psi
    print("\nRunning pbh_merger_rate_callable.py example...")

    # --- Define psi for example (Bimodal Gaussian) ---
    m1_peak_ex = 1.0; m2_peak_ex = 1.0e-4; sigma_rel_ex = 0.01
    sigma_g1_ex = max(sigma_rel_ex * m1_peak_ex, 1e-12 * m1_peak_ex)
    sigma_g2_ex = max(sigma_rel_ex * m2_peak_ex, 1e-12 * m2_peak_ex)
    def mass_func_psi_norm_ex(m):
        pdf1 = 0.5 * scipy.stats.norm.pdf(m, loc=m1_peak_ex, scale=sigma_g1_ex)
        pdf2 = 0.5 * scipy.stats.norm.pdf(m, loc=m2_peak_ex, scale=sigma_g2_ex)
        result = pdf1 + pdf2
        return max(0.0, result) if np.isscalar(m) else np.maximum(0.0, result)

    m_min_psi_ex = m2_peak_ex / 1000; m_max_psi_ex = m1_peak_ex * 10
    num_points_psi_ex = 1000
    mass_grid_psi_ex = np.logspace(np.log10(m_min_psi_ex), np.log10(m_max_psi_ex), num_points_psi_ex)
    psi_vals_grid_ex = mass_func_psi_norm_ex(mass_grid_psi_ex)
    norm_check_ex = scipy.integrate.trapz(psi_vals_grid_ex, mass_grid_psi_ex)
    print(f"Example psi normalization check: {norm_check_ex:.4g}")
    # Create interpolator for the example psi
    unique_m_ex, unique_idx_ex = np.unique(mass_grid_psi_ex, return_index=True)
    if len(unique_m_ex) < 2: raise ValueError("Example grid issue?")
    psi_interp_ex = scipy.interpolate.interp1d(unique_m_ex, psi_vals_grid_ex[unique_idx_ex], bounds_error=False, fill_value=0.0)

    # --- Run calculation ---
    test_f_total = 0.001
    test_mlist = np.logspace(-5, 2, 50) # Output grid

    mass_out, rate_mat_out, fpbh0_out, total_rate_out = calculate_merger_rate_matrix(
        f_total=test_f_total,
        psi_interp_func=psi_interp_ex, # Pass interpolator
        mlist=test_mlist,
        m1_peak=m1_peak_ex, # Pass peak masses for analytical moments
        m2_peak=m2_peak_ex,
        use_analytical_moments=True # Use analytical moments for this sharp case
    )

    print(f"\nExample results for f={test_f_total}:")
    # Note: fpbh0_out is just f_total now
    print(f"Total Integrated Rate = {total_rate_out:.4g} Gpc^-3 yr^-1")
    idx1_ex = np.argmin(np.abs(test_mlist - m1_peak_ex))
    idx2_ex = np.argmin(np.abs(test_mlist - m2_peak_ex))
    print(f"Rate Density M1-M1: {rate_mat_out[idx1_ex, idx1_ex]:.3g}")
    print(f"Rate Density M2-M2: {rate_mat_out[idx2_ex, idx2_ex]:.3g}")
    print(f"Rate Density M1-M2: {rate_mat_out[idx1_ex, idx2_ex]:.3g}")


    # --- Save example output ---
    output_dir_example = "."
    output_file_example = os.path.join(output_dir_example, f"pbh_merger_rate_example_f{test_f_total:.1e}.npz")
    np.savez(output_file_example, m=mass_out, rate=rate_mat_out, fpbh=test_f_total, total_rate=total_rate_out)
    print(f"Example PBH merger rate saved to {output_file_example}")
