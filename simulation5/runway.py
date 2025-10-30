import numpy as np
import scipy.integrate
import scipy.special
from scipy.misc import derivative # For d(sigma)/d(Ni)

# --- Constants ---
G_N = ... # Newton's constant in appropriate units (e.g., matching the paper)
# ... other constants like kappa, g_*,H, delta_c, Delta=18pi^2 ...
rho_DM_today = ... # Dark matter density today (e.g., in g/cm^3)

# --- Input Parameters ---
# Examples from Fig 4 [cite: 662]
# mi = 1e9 # Initial PBH mass [g]
# ti = 1e-20 # Time BHD begins [s]
mi = 1e6 # g
ti = 1e-30 # s

# --- Helper Functions (Implement based on paper's equations) ---

def hawking_lifetime(m, g_star_H=108, kappa=3.8):
    """Calculates PBH lifetime tau based on Eq. (5)[cite: 125]."""
    # Check units carefully
    tau = (10240 * np.pi * G_N**2 * m**3) / (kappa * g_star_H)
    return tau # in seconds

def hawking_mass_evol(m_init, t_init, t_final, tau_init):
    """Calculates mass at t_final given mass m_init at t_init. Eq. (6) [cite: 128]."""
    if t_final <= t_init: return m_init
    if t_final >= t_init + tau_init: return 0.0 # Fully evaporated
    # Assuming t_form is negligible compared to t_init
    time_elapsed_since_form = t_final # Approximation
    time_remaining_ratio = 1.0 - time_elapsed_since_form / tau_init
    if time_remaining_ratio < 0: return 0.0
    return m_init * (time_remaining_ratio)**(1/3)

def horizon_mass(t):
    """Horizon mass M_H based on Eq. (10) [cite: 155]."""
    # Check definition and units
    M_H = (27 * np.pi * t) / (8 * G_N) # Needs factor of c? Check paper units.
    return M_H # in grams

def NH_i(ti, mi):
    """Initial PBH number in horizon mass. Eq. (24) [cite: 227]."""
    MH_i = horizon_mass(ti)
    if mi <= 0: return np.inf
    return MH_i / mi

def cluster_mu(Ni, NH_i_val):
    """Cluster mass transfer function mu(N_i) based on Eq. (21) [cite: 212]."""
    if Ni < NH_i_val:
        return 1.0 - (4.0/7.0) * (Ni / NH_i_val)
    else:
        return (3.0/7.0) * (Ni / NH_i_val)**(-4.0/3.0)

def sigma_M_sq(Ni, mi, ti, t, NH_i_val):
    """Variance sigma_M^2 based on Eq. (20) [cite: 210]."""
    if Ni <= 0 or mi <= 0 or ti <=0: return 0.0
    M = Ni * mi
    mu_i = cluster_mu(Ni, NH_i_val)
    D_a = (t / ti)**(2.0/3.0) # Growth factor D(a) = a/ai = (t/ti)^(2/3)
    # Check the factor mi/M = 1/Ni
    return (1.0 / Ni) * mu_i * D_a**2

def dsigmaM_dNi_numeric(Ni, mi, ti, t, NH_i_val):
     """Numerically calculates d(sigma_M)/d(Ni)."""
     sigma_func = lambda n: np.sqrt(sigma_M_sq(n, mi, ti, t, NH_i_val))
     # Using scipy.misc.derivative might require adjustments or an alternative
     try:
         deriv = derivative(sigma_func, Ni, dx=Ni*1e-6, n=1)
     except Exception:
         # Fallback finite difference
         h = Ni * 1e-6
         deriv = (sigma_func(Ni + h) - sigma_func(Ni - h)) / (2 * h)
     return deriv

def press_schechter_dndlni(Ni, mi, ti, t, delta_c=1.686):
    """Press-Schechter dn_cl / d(ln Ni) based on Eq. (23) [cite: 224]."""
    # Note: Eq (23) is dn/dNi. Need to multiply by Ni for dn/dlnNi
    if Ni <= 0: return 0.0
    NH_i_val = NH_i(ti, mi)
    mu_i = cluster_mu(Ni, NH_i_val)
    if mu_i <= 0: return 0.0
    D = (t / ti)**(2.0/3.0)
    sigma_sq = sigma_M_sq(Ni, mi, ti, t, NH_i_val)
    sigma = np.sqrt(sigma_sq)

    # Need d(log mu)/d(log Ni)
    h_log = 1e-6
    log_mu_func = lambda logN: np.log(cluster_mu(np.exp(logN), NH_i_val))
    try:
        dlogmu_dlogN = derivative(log_mu_func, np.log(Ni), dx=h_log, n=1)
    except Exception:
         dlogmu_dlogN = (log_mu_func(np.log(Ni)*(1+h_log)) - log_mu_func(np.log(Ni)*(1-h_log))) / (2*np.log(Ni)*h_log)

    term1 = delta_c / (np.sqrt(2 * np.pi * mu_i * Ni) * D) # Adjusted factor Ni^3 -> Ni
    term2 = (1.0 - dlogmu_dlogN)
    exp_term = np.exp(- (Ni * delta_c**2) / (2 * mu_i * D**2))

    dn_dNi = term1 * term2 * exp_term # This is dn/dNi per unit *initial* comoving volume *per BH*
    # Convert to dn/dlnNi
    dn_dlnNi = Ni * dn_dNi
    
    # Need number density n_BH_initial (at ti) to normalize
    rho_BH_i = 1.0 / (6 * np.pi * G_N * ti**2) # Eq (8) [cite: 146]
    n_BH_i = rho_BH_i / mi
    
    return n_BH_i * dn_dlnNi # This is now comoving number density dn_cl / dlnNi

def cluster_formation_time(Ni, ti, mi, delta_c=1.686):
    """Cluster formation time t_cl based on Eq. (25)[cite: 235]."""
    NH_i_val = NH_i(ti, mi)
    mu_i = cluster_mu(Ni, NH_i_val)
    if mu_i <= 0: return np.inf
    t_cl = (delta_c**(3.0/2.0)) * (Ni / mu_i)**(3.0/4.0) * ti
    return t_cl

def initial_cluster_rho(Ni, ti, mi, delta_c=1.686, Delta=18*np.pi**2):
    """Initial cluster density rho_cl,i based on Eq. (26) [cite: 241]."""
    NH_i_val = NH_i(ti, mi)
    mu_i = cluster_mu(Ni, NH_i_val)
    if mu_i <= 0: return 0.0
    rho_bh_ti = 1.0 / (6 * np.pi * G_N * ti**2) # Avg density at ti
    a_cl_over_a_i = delta_c * np.sqrt(Ni / mu_i)
    rho_bh_tcl = rho_bh_ti * (a_cl_over_a_i)**-3
    rho_cli = Delta * rho_bh_tcl
    # Formula in paper seems simplified, let's use it:
    # rho_cli = (Delta / (6 * np.pi * G_N * ti**2 * delta_c**3)) * (mu_i / Ni)**(3.0/2.0)
    return rho_cli # Density in g/cm^3 or similar

def initial_cluster_R(Ni, mi, rho_cl_i):
    """Initial cluster radius R_i based on Eq. (27) [cite: 248]."""
    M_i = Ni * mi
    if rho_cl_i <= 0: return np.inf
    R_i = ( (3 * M_i) / (4 * np.pi * rho_cl_i) )**(1.0/3.0)
    return R_i # in cm or pc

def initial_cluster_sigma_v(Ni, mi, R_i):
    """Initial velocity dispersion sigma_v,i based on Eq. (28) [cite: 251]."""
    M_i = Ni * mi
    if R_i <= 0: return 0.0
    sigma_v_i_sq = (G_N * M_i) / R_i
    return np.sqrt(sigma_v_i_sq) # Velocity units

def initial_gamma_mrg(Ni, mi, R_i, sigma_v_i):
    """Initial 2-body merger rate Gamma_mrg,i using Eq. (37)  and helpers."""
    if Ni <=0 or mi <=0 or R_i <=0 or sigma_v_i <=0: return 0.0
    M_i = Ni * mi
    rho_cl_i = (3 * M_i) / (4 * np.pi * R_i**3)
    n_BH_i = rho_cl_i / mi
    v_rel = (4.0 / np.sqrt(np.pi)) * sigma_v_i # Eq. (32) text [cite: 290]
    r_s = 2 * G_N * mi # Schwarzschild radius for component mass
    # Sigma_2b based on Eq (31) [cite: 282]
    sigma_2b = ((85*np.pi/3)**(2.0/7.0)) * np.pi * r_s**2 / v_rel**(18.0/7.0)
    gamma_mrg = 0.5 * Ni * n_BH_i * sigma_2b * v_rel
    return gamma_mrg # Rate in 1/s

def initial_gamma_ev(Ni, R_i, sigma_v_i, gamma_ev_const=7.4e-3):
    """Initial cluster evaporation rate Gamma_ev,i using Eq. (39)  and helpers."""
    if Ni <= 1 or sigma_v_i <= 0: return 0.0 # Evaporation requires N>1
    logN = np.log(Ni)
    if logN <= 0: logN = np.log(1.1) # Avoid log(1)
    t_dyn = R_i / sigma_v_i # Eq (38) text [cite: 319]
    t_rlx = (Ni / (8.0 * logN)) * t_dyn # Eq (38) [cite: 322]
    if t_rlx <= 0: return np.inf
    gamma_ev = (gamma_ev_const * Ni) / t_rlx
    return gamma_ev # Rate in 1/s

def t_mrg_runaway(Ni, gamma_mrg_i):
    """Runaway merger timescale based on Eq. (44)[cite: 366]."""
    if gamma_mrg_i <= 0: return np.inf
    return Ni / gamma_mrg_i

def t_ev_runaway(Ni, gamma_ev_i):
    """Cluster evaporation timescale based on Eq. (49) [cite: 400]."""
    if gamma_ev_i <= 0: return np.inf
    return (2.0 * Ni) / (7.0 * gamma_ev_i)

def t_collapse(t_mrg_r, t_ev_r):
    """Collapse timescale based on Eq. (54) ."""
    return min(t_mrg_r, t_ev_r)

def m_relic_final(Ni, mi, gamma_mrg_i, gamma_ev_i, delta_m_frac=0.05):
    """Final relic mass based on Eq. (53) ."""
    if Ni <= 1: return mi # Single PBH doesn't merge
    if gamma_ev_i <= 0: # Merger dominated from start
         # Eq (53) gives m_relic -> infinity if gamma_ev=0. Use total mass.
         return Ni * mi * (1 - delta_m_frac) # Approx total mass minus GW loss
    ratio_gamma = gamma_mrg_i / gamma_ev_i
    term1 = 1.0 + ratio_gamma * Ni**(5.0/14.0)
    term2 = 1.0 + ratio_gamma
    if term1 <= 0 or term2 <= 0: return mi # Avoid errors
    exponent = (14.0/25.0) * (1.0 - delta_m_frac)
    m_rel = mi * (term1 / term2)**exponent
    # Ensure relic mass is at least initial mass
    return max(mi, m_rel)

def N_collapse_max(tau_val, ti, mi):
    """Find max Ni such that t_collapse(Ni) <= tau_val. Needs root finding."""
    # Define function whose root is needed: f(Ni) = t_collapse(Ni) - tau_val
    def collapse_time_func(log10_Ni):
        Ni_val = 10**log10_Ni
        if Ni_val <= 1: return np.inf # Avoid Ni=1 issues
        rho_cl_i = initial_cluster_rho(Ni_val, ti, mi)
        R_i = initial_cluster_R(Ni_val, mi, rho_cl_i)
        sigma_v_i = initial_cluster_sigma_v(Ni_val, mi, R_i)
        gamma_mrg_i = initial_gamma_mrg(Ni_val, mi, R_i, sigma_v_i)
        gamma_ev_i = initial_gamma_ev(Ni_val, R_i, sigma_v_i)
        t_mrg_r = t_mrg_runaway(Ni_val, gamma_mrg_i)
        t_ev_r = t_ev_runaway(Ni_val, gamma_ev_i)
        t_col = t_collapse(t_mrg_r, t_ev_r)
        return t_col - tau_val

    # Use a root finder (e.g., scipy.optimize.brentq)
    from scipy.optimize import brentq
    try:
        # Need appropriate bounds for root finding
        log10_Ni_min = 0.1 # Min Ni > 1
        log10_Ni_max = 30 # Max Ni guess (needs adjustment based on params)
        log10_Nmax = brentq(collapse_time_func, log10_Ni_min, log10_Ni_max, xtol=1e-3, rtol=1e-3)
        return 10**log10_Nmax
    except ValueError:
        print(f"Warning: Root finding for N_col_max failed. Check bounds/function.")
        # Check if collapse time is always > tau or always < tau
        if collapse_time_func(log10_Ni_min) > 0: return 1.0 # No clusters collapse
        if collapse_time_func(log10_Ni_max) < 0: return 10**log10_Ni_max # All clusters collapse?
        return 1.0 # Fallback

def get_relic_mass_distribution_at_tau(Ni_array, mi, ti, tau_val, N_col_max_val):
    """Calculates dn/dm(tau) using Eq. (57) ."""
    dn_dlogm = {} # Dictionary to store distribution binned by log(m)

    m_relic_vals = np.zeros_like(Ni_array)
    dmdNi_vals = np.zeros_like(Ni_array)

    # Calculate m_relic and its derivative for relevant Ni range
    Ni_merging = Ni_array[Ni_array < N_col_max_val]
    for i, Ni_val in enumerate(Ni_merging):
        rho_cl_i = initial_cluster_rho(Ni_val, ti, mi)
        R_i = initial_cluster_R(Ni_val, mi, rho_cl_i)
        sigma_v_i = initial_cluster_sigma_v(Ni_val, mi, R_i)
        gamma_mrg_i = initial_gamma_mrg(Ni_val, mi, R_i, sigma_v_i)
        gamma_ev_i = initial_gamma_ev(Ni_val, R_i, sigma_v_i)
        m_relic_vals[i] = m_relic_final(Ni_val, mi, gamma_mrg_i, gamma_ev_i)

        # Numerical derivative dm_relic / dNi
        h = Ni_val * 1e-6
        m_relic_plus = m_relic_final(Ni_val + h, mi, gamma_mrg_i, gamma_ev_i) # Approx rates same
        m_relic_minus = m_relic_final(Ni_val - h, mi, gamma_mrg_i, gamma_ev_i)
        dmdNi_vals[i] = (m_relic_plus - m_relic_minus) / (2 * h)

    # Calculate Press-Schechter at tau
    dn_dlnNi_at_tau = np.array([press_schechter_dndlni(Ni_val, mi, ti, tau_val) for Ni_val in Ni_array])

    # Bin the results by log(m_relic) - Jacobian transformation
    num_bins = 100
    min_logm = np.log10(mi)
    max_logm = np.log10(m_relic_vals[Ni_merging > 1].max()) if np.any(Ni_merging > 1) else min_logm + 1
    logm_bins = np.linspace(min_logm, max_logm, num_bins + 1)
    logm_centers = 0.5 * (logm_bins[:-1] + logm_bins[1:])
    dndlogm_binned = np.zeros_like(logm_centers)

    for i, Ni_val in enumerate(Ni_merging):
        if Ni_val <= 1 or dmdNi_vals[i] <= 0: continue # Skip single PBHs or issues
        m_rel_i = m_relic_vals[i]
        logm_rel_i = np.log10(m_rel_i)
        
        # Find which bin this relic mass falls into
        bin_idx = np.digitize(logm_rel_i, logm_bins) - 1
        
        if 0 <= bin_idx < num_bins:
            # Jacobian: dn/dlogm = dn/dNi * dNi/dlogm = (dn/dlnNi) * (dlnNi / dlogm)
            # dlogm / dlnNi = (dlogm / dm) * (dm / dNi) * (dNi / dlnNi)
            # = (1 / (m ln10)) * (dm/dNi) * Ni
            dlogm_dlnNi = (1.0 / (m_rel_i * np.log(10))) * dmdNi_vals[i] * Ni_val
            if dlogm_dlnNi > 0:
                 jacobian_factor = 1.0 / dlogm_dlnNi
                 # Find corresponding dn/dlnNi value (may need interpolation if Ni_array is coarse)
                 dn_dlnNi_i = np.interp(Ni_val, Ni_array, dn_dlnNi_at_tau)
                 dndlogm_binned[bin_idx] += dn_dlnNi_i * jacobian_factor
                 
    # Add the non-merging PBHs (delta function approximation)
    # A proper treatment integrates PS from N_col_max up
    rho_BH_tau = 1.0 / (6 * np.pi * G_N * tau_val**2) # Density at tau
    n_BH_tau = rho_BH_tau / mi # Approx number density (ignoring mass loss slightly)
    # Find bin for mi
    mi_bin_idx = np.digitize(np.log10(mi), logm_bins) - 1
    if 0 <= mi_bin_idx < num_bins:
        # Approximate delta function by putting all density in one bin
        # This isn't quite right, needs careful normalization check
        # The paper approximates this term as n_BH(tau) * delta(m-mi) [cite: 466]
        # For binning, we approximate delta(m-mi) contribution
         dndlogm_binned[mi_bin_idx] += n_BH_tau # Add density here? Needs check.

    return logm_centers, dndlogm_binned # dn/dlogm [cm^-3] or similar

def convert_dndlogm_to_dfdlogm(logm, dndlogm, rho_DM_today_val):
    """Converts dn/dlogm today to df_BH/dlogm today using Eq. (59)."""
    m = 10**logm
    # Need to redshift number density from tau to today
    # T_RH = T_reheat(mi) # Eq (15) 
    # a_tau / a_0 ~ T_0 / T_RH (approx, needs entropy conservation)
    # n_today = n_tau * (a_tau / a_0)^3
    # Placeholder: Assume a fixed redshift factor (INCORRECT - NEEDS PROPER COSMOLOGY)
    redshift_factor_cubed = 1e-30 # Highly dependent on T_RH -> mi
    
    dndlogm_today = dndlogm # * redshift_factor_cubed # Apply redshift factor
    
    rho_BH_dist = m * dndlogm_today * np.log(10) # rho = m * dn = m * (dn/dlogm) * dlogm = m*dn/dlogm*ln10
    
    dfdlogm = rho_BH_dist / rho_DM_today_val
    return dfdlogm


# --- Main Calculation ---

print(f"Calculating for mi={mi:.1e} g, ti={ti:.1e} s")
tau = hawking_lifetime(mi)
print(f"  PBH lifetime tau = {tau:.3g} s")

# Check viability conditions
if ti >= tau:
    print("  ERROR: PBHs evaporate before domination starts (ti > tau).")
    exit()
if mi >= horizon_mass(ti):
    print("  ERROR: PBH mass exceeds horizon mass at ti.")
    exit()
# BBN check on reheat temperature would go here (using Eq 15 )

N_col_max = N_collapse_max(tau, ti, mi)
print(f"  Max cluster size collapsing by tau: N_col_max = {N_col_max:.3g}")

# Define Ni grid for integration
Ni_min_calc = 1.0
Ni_max_calc = max(N_col_max * 10, NH_i(ti, mi)*100) # Ensure grid covers relevant scales
Ni_grid = np.logspace(np.log10(Ni_min_calc), np.log10(Ni_max_calc), 500)

# Calculate mass distribution at tau
logm_centers_tau, dndlogm_tau = get_relic_mass_distribution_at_tau(Ni_grid, mi, ti, tau, N_col_max)

# Evolve mass distribution to today
logm_centers_today = logm_centers_tau
dndlogm_today = np.zeros_like(dndlogm_tau)
masses_at_tau = 10**logm_centers_tau
t_today_approx = 13.8e9 * 3.154e7 # Age of universe in seconds

for i, m_at_tau in enumerate(masses_at_tau):
    if dndlogm_tau[i] <= 0: continue
    tau_relic = hawking_lifetime(m_at_tau)
    m_today = hawking_mass_evol(m_at_tau, tau, t_today_approx, tau_relic)
    
    if m_today > 0:
        # Find corresponding bin for m_today (needs re-binning or interpolation)
        # Simple shift approximation (assumes logm bins stay roughly same):
        dndlogm_today[i] = dndlogm_tau[i] # Density remains in 'comoving' bin
        logm_centers_today[i] = np.log10(m_today) # Update the mass label of the bin

# Convert to df/dlogm
dfdlogm_today = convert_dndlogm_to_dfdlogm(logm_centers_today, dndlogm_today, rho_DM_today)

# --- Plotting (Simplified) ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Plot initial spike (approx)
ax.axvline(mi, color='green', linestyle=':', label='Initial $m_i$', lw=2)

# Plot relic distribution at tau (Needs conversion to df/dlogm at tau)
# dfdlogm_tau = convert_dndlogm_to_dfdlogm(logm_centers_tau, dndlogm_tau, rho_DM_today) # Need correct redshift
# ax.plot(10**logm_centers_tau, dfdlogm_tau, color='blue', label='Relics at $\\tau$')

# Plot relic distribution today
mask_today = dfdlogm_today > 0
ax.plot(10**logm_centers_today[mask_today], dfdlogm_today[mask_today], color='red', label='Relics Today ($t_0$)')

# --- Add Constraints (Requires loading external data) ---
# Example: Placeholder for BBN constraint
# ax.fill_between([1e9, 1e13], 1e-8, 1e2, color='cyan', alpha=0.3, label='BBN Constraint (Example)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('BH mass m [g]')
ax.set_ylabel('$df_{BH}/d\log m$')
ax.set_title(f'Relic Mass Distribution for $m_i={mi:.1e}$ g, $t_i={ti:.1e}$ s')
ax.legend()
ax.grid(True, which='both', ls=':')
plt.ylim(1e-10, 1e2) # Match Fig 4 y-limits approx
plt.xlim(10**(np.log10(mi)-1), 1e17) # Adjust x-limits
plt.tight_layout()
plt.savefig(f"relic_mass_dist_mi_{mi:.1e}_ti_{ti:.1e}.png")
print(f"Plot saved to relic_mass_dist_mi_{mi:.1e}_ti_{ti:.1e}.png")
