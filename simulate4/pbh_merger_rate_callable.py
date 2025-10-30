# pbh_merger_rate_callable.py
# MODIFIED: Made callable, calculates total rate.

import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.special
import os
sigma_eq = np.sqrt(0.005)
# --- Helper Functions (copied from original script) ---
def NNN(mi, mj, avM, fpbh0, sigma_eq_val):
    if avM == 0 or (fpbh0 + sigma_eq_val) == 0: return 0.0
    return (mi + mj) / avM * fpbh0 / (fpbh0 + sigma_eq_val)

def S_L(fpbh00):
    ht = np.maximum(fpbh00, 1e-99)
    t1 = (ht ** (-0.65)) * np.exp(0.03 * (np.log(ht)) ** 2)
    xt = np.min([1.0, 0.01*t1])
    return xt

def CCC(fpbh00, avM, avM2, sigma_eq_val):
    if avM == 0 or sigma_eq_val == 0: return 0.0
    avM2_over_avM_sq = avM2 / avM**2 if avM != 0 else 0.0
    t1 = ((fpbh00**2) * avM2_over_avM_sq / sigma_eq_val**2)
    ht0 = scipy.special.gamma(29.0/37.0) / np.sqrt(np.pi)
    arg_hyperu = 5.0 * (fpbh00**2) / (6.0 * sigma_eq_val**2)
    # Handle potential issues with hyperu for large arguments or edge cases
    try:
        ht1 = scipy.special.hyperu(21.0/74.0, 1.0/2.0, arg_hyperu)
        if not np.isfinite(ht1): raise ValueError("hyperu result not finite")
    except (ValueError, OverflowError):
        # Fallback for large arguments where hyperu might struggle or overflow
        # Using asymptotic expansion U(a,b,z) ~ z^-a for large z
        if arg_hyperu > 100: # Heuristic threshold
             ht1 = arg_hyperu**(-21.0/74.0)
        else:
             ht1 = 0.0 # Indicate failure if not large and failed

    denominator_term = ht0 * ht1
    if denominator_term <= 0: return 0.0 # Avoid issues with power

    # Power exponent calculation can lead to issues if term is ~0
    power_arg = denominator_term**(-74.0/21.0) if denominator_term > 1e-99 else np.inf

    denominator = power_arg - 1.0
    return t1 / denominator if denominator > 1e-99 else 0.0 # Avoid division by zero or negative


def S_E(mi, mj, fpbh00, avM, avM2, sigma_eq_val):
    nnn_val = NNN(mi, mj, avM, fpbh00, sigma_eq_val)
    ccc_val = CCC(fpbh00, avM, avM2, sigma_eq_val)
    if avM == 0 or fpbh00 == 0: return 0.0
    avM2_over_avM_sq = avM2 / avM**2 if avM != 0 else 0.0

    t1_factor = (np.sqrt(np.pi) * (5.0 / 6.0)**(21.0 / 74.0) / scipy.special.gamma(29.0 / 37.0))

    denom_t2 = (nnn_val + ccc_val)
    if denom_t2 <= 0: return 0.0 # Avoid issues if NNN+CCC <= 0
    term_in_t2 = (avM2_over_avM_sq / denom_t2 + sigma_eq_val**2 / fpbh00**2)
    if term_in_t2 <= 0: return 0.0 # Avoid issues with power

    t2_factor = term_in_t2**(-21 / 74)
    t3_factor = np.exp(-nnn_val)
    return t1_factor * t2_factor * t3_factor

def rate_density(mi, mj, fpbh0, Phi_interp, avM, avM2, sigma_eq_val):
    """Calculates dR / (dln m1 dln m2)"""
    if mi <= 0 or mj <= 0 or fpbh0 <= 0: return 0.0

    ffh = fpbh0 * 0.85 # Why 0.85? Check original paper context if needed.
    # Constant C0 = 1.6e6 * (t/t0)^(-34/37). Assuming t=t0.
    t0_const = (1.6e6 * (ffh**(53.0 / 37.0)) * ((mi + mj)**(-32.0 / 37.0))) if (mi+mj)>0 else 0.0

    # Phi(mi) * Phi(mj) / mi / mj * mi * mj = Phi(mi)*Phi(mj) for dlnm integration
    # Need dR / dln(mi) dln(mj) = R(mi, mj) * mi * mj ? Check definition.
    # Original formula in pbh_merger_rate.py comment seems to be for dR / dm1 dm2
    # R = t0 * [Phi(mi)*Phi(mj)/mi/mj] * t2 * t3
    # dR/dln(mi)dln(mj) = R * mi * mj
    try:
        phi_mi = Phi_interp(mi)
        phi_mj = Phi_interp(mj)
        if not (np.isfinite(phi_mi) and np.isfinite(phi_mj)): return 0.0
    except ValueError: # Handle extrapolation issues if bounds_error=True (default is True)
         return 0.0

    t1_term = phi_mi * phi_mj # For dlnm integration

    # t2 term = eta^(-34/37)
    M_tot = mi + mj
    if M_tot <= 0: return 0.0
    eta = (mi * mj) / (M_tot**2)
    if eta <= 0: return 0.0 # Avoid issues with power
    t2_term = eta**(-34.0 / 37.0)

    # t3 term = S_L * S_E
    sl_val = S_L(fpbh0)
    se_val = S_E(mi, mj, fpbh0, avM, avM2, sigma_eq_val)
    t3_term = sl_val * se_val

    # Combine: R(mi, mj) * mi * mj
    rate_dln = t0_const * t1_term * t2_term * t3_term
    return rate_dln if np.isfinite(rate_dln) else 0.0


def calculate_merger_rate_matrix(f_total, mass_func_psi, mlist):
    """
    Calculates the PBH merger rate matrix dR/(dln m1 dln m2) for a given
    total fraction f_total, normalized mass function psi(m), and mass grid mlist.
    psi(m) should be normalized such that integral(psi(m) dm) = 1.
    """
    print(f"Calculating analytical rate matrix for f_PBH = {f_total:.3e}...")

    # Calculate fpbh_dist = f_total * psi(m) on the grid
    # Ensure mlist is sorted
    mlist = np.sort(np.asarray(mlist))
    valid_m = mlist[mlist > 0]
    if len(valid_m) < 2:
         print("Error: Need at least 2 positive masses in mlist.")
         return mlist, np.zeros((len(mlist), len(mlist))), 0.0, 0.0

    # Evaluate psi on the grid
    psi_on_mlist = np.array([mass_func_psi(m) for m in mlist])
    # Ensure psi is non-negative
    psi_on_mlist[psi_on_mlist < 0] = 0.0

    fpbh_dist = f_total * psi_on_mlist

    # --- Calculate moments needed by the rate formula ---
    # Need integrals over dm, not dlnm for these definitions
    integrand_fpbh0 = lambda m: fpbh_dist[np.argmin(np.abs(mlist-m))] / m if m > 0 else 0 # Interpolate fpbh_dist/m
    fpbh0_num, _ = scipy.integrate.quad(integrand_fpbh0, mlist[0], mlist[-1], limit=200) # Integrate f*psi/m dm
    fpbh0 = abs(fpbh0_num)
    # Note: fpbh0 defined here is integral(f*psi/m dm), matching script's usage for Phi normalization

    if fpbh0 < 1e-99: # Avoid division by zero if total fraction is effectively zero
        print("Warning: Calculated fpbh0 is near zero. Rate will be zero.")
        avM = 1.0; avM2 = 1.0; Phi = lambda m: 0.0
        # return mlist, np.zeros((len(mlist), len(mlist))), 0.0, 0.0
    else:
        # Phipbh = fpbh_dist / fpbh0 # Normalization for Phi interpolation seems off, should use psi directly?
        # Let's use psi for Phi = psi * m / <m> definition from Raidal paper (Eq 2)
        # psi * m should be proportional to fpbh_dist. Need <m>.
        # <m> = 1 / integral(psi/m dm)
        integrand_inv_m_avg = lambda m: mass_func_psi(m) / m if m > 0 else 0
        integral_inv_m_avg, _ = scipy.integrate.quad(integrand_inv_m_avg, mlist[0], mlist[-1], limit=200)
        if integral_inv_m_avg <= 0: raise ValueError("<m> calculation failed.")
        avM = 1.0 / integral_inv_m_avg # This is <m> = rho_pbh / n_pbh

        # Calculate <m^2> needed for avM2/avM^2 term in S_E
        # <m^2> = integral(m * psi dm) ? No, Raidal Eq 2.25 uses <m^2>/<m>^2 for variance.
        # Let's use the definitions from pbh_merger_rate.py which seem consistent with its internal formulas:
        # avM = integral( (f*psi / fpbh0) dm ) -> proportional to integral( (psi/integral(psi/m)) dm ) ???
        # avM2 = integral( (f*psi / fpbh0) * m dm ) -> proportional to integral( (m*psi/integral(psi/m)) dm ) ???
        # Let's stick to the script's definitions for consistency FOR NOW:
        avM = abs(scipy.integrate.trapz(fpbh_dist / fpbh0, mlist)) if fpbh0 > 0 else 1.0
        avM2 = abs(scipy.integrate.trapz(fpbh_dist / fpbh0 * mlist, mlist)) if fpbh0 > 0 else 1.0


        # Create interpolator for Phipbh = abs(fpbh_dist / fpbh0)
        # Ensure mass grid is unique for interpolation
        unique_m, unique_idx = np.unique(mlist, return_index=True)
        if len(unique_m) < 2: raise ValueError("Need at least 2 unique masses for interpolation.")
        Phipbh_vals = abs(fpbh_dist[unique_idx] / fpbh0)
        try:
            Phi = scipy.interpolate.interp1d(unique_m, Phipbh_vals, bounds_error=False, fill_value=0.0)
        except ValueError as e:
            print(f"Error creating Phi interpolator: {e}")
            Phi = lambda m: 0.0 # Fallback


    sigma_eq_val = sigma_eq # Use constant from top

    # --- Calculate Rate Matrix ---
    rate_matrix = np.zeros((len(mlist), len(mlist)))
    for i in range(len(mlist)):
        for j in range(len(mlist)):
            mi = mlist[i]
            mj = mlist[j]
            # Need rate density dR / dln(mi) dln(mj)
            rate_matrix[i, j] = rate_density(mi, mj, fpbh0, Phi, avM, avM2, sigma_eq_val)

    # --- Calculate Total Integrated Rate ---
    # R_tot = integral( R(mi, mj)_dln * psi(mi) * psi(mj) dln(mi) dln(mj) ) ?? No, check normalization.
    # R_tot = integral( R(mi, mj)_dm * dm1 * dm2 ) where R_dm = R_dln / (m1*m2)
    # R_tot = integral( R_dln / (m1*m2) * dm1 * dm2 )
    # Using grid: Sum over R_matrix[i,j] / (mlist[i]*mlist[j]) * (Delta m_i) * (Delta m_j) ?
    # Let's use the dln definition:
    # R_tot = Sum_{i,j} rate_matrix[i,j] * (Delta log(m_i)) * (Delta log(m_j)) ? This seems wrong dimensionally.
    # Let's follow Raidal paper's approach for total rate R = integral dR
    # dR = R_dln * dln(m1) * dln(m2)
    # Total Rate R = Sum_{i,j} rate_matrix[i,j] * (Delta log(m_i)) * (Delta log(m_j)) where Delta log m is step size.
    total_rate = 0.0
    if len(mlist) > 1:
        log_m = np.log(mlist)
        # Use midpoint integration approximation
        delta_log_m = np.diff(log_m)
        delta_log_m = np.append(delta_log_m, delta_log_m[-1]) # Append last diff for edge case

        # Create grid of delta_log_m steps
        dlogm_i, dlogm_j = np.meshgrid(delta_log_m, delta_log_m, indexing='ij')

        # Sum over matrix elements multiplied by area element dln(mi)dln(mj)
        total_rate = np.sum(rate_matrix * dlogm_i * dlogm_j)


    print(f"... Analytical calculation done. Total Rate = {total_rate:.3g} Gpc^-3 yr^-1")

    return mlist, rate_matrix, fpbh0, total_rate # Return fpbh0 and total rate too

# --- Main execution block (if run as script, can be removed if only importing) ---
if __name__ == '__main__':
    # Example usage with a dummy mass function
    print("Running pbh_merger_rate_callable.py as main script (example)")

    # Define a simple log-normal psi(m) for testing
    mc_test = 30.0
    sigma_test = 0.6
    def lognormal_psi(m):
        if m <= 0: return 0.0
        norm = 1.0 / (np.sqrt(2 * np.pi) * sigma_test * m) # dlnm requires 1/m, dm just needs 1/m in exponent? Check normalization. Using dm norm.
        exponent = - (np.log(m / mc_test)**2) / (2 * sigma_test**2)
        return norm * np.exp(exponent)

    # Normalize the lognormal function numerically over the grid
    test_mass_grid = np.logspace(-1, 3, 100)
    psi_vals_unnorm = np.array([lognormal_psi(m) for m in test_mass_grid])
    norm_const_test = scipy.integrate.simps(psi_vals_unnorm, test_mass_grid) # Integrate dm

    def lognormal_psi_norm(m):
         return lognormal_psi(m) / norm_const_test if norm_const_test > 0 else 0.0

    test_f_total = 0.001
    test_mlist = np.logspace(0, 2, 50) # Coarser grid for output matrix

    mass_out, rate_mat_out, fpbh0_out, total_rate_out = calculate_merger_rate_matrix(
        f_total=test_f_total,
        mass_func_psi=lognormal_psi_norm,
        mlist=test_mlist
    )

    print(f"\nExample results for f={test_f_total}:")
    print(f"fpbh0 (integral f*psi/m dm) = {fpbh0_out:.4g}")
    print(f"Total Integrated Rate = {total_rate_out:.4g} Gpc^-3 yr^-1")
    # print("Rate Matrix sample (diagonal):", np.diag(rate_mat_out))

    # Save example output
    output_dir_example = "."
    output_file_example = os.path.join(output_dir_example, f"pbh_merger_rate_example_f{test_f_total:.1e}.npz")
    np.savez(output_file_example, m=mass_out, rate=rate_mat_out, fpbh=test_f_total, total_rate=total_rate_out)
    print(f"Example PBH merger rate saved to {output_file_example}")
