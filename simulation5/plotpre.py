# plot_pdf_comparison.py
#
# Loads a precomputed P(a,j) grid from a .npz file and plots it
# against the simplified analytical formula (Raidal Eq. 2.32) for comparison.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
import os

# --- Plotting Style ---
mpl.rcParams.update({
    'font.size': 14, 'font.family': 'sans-serif', 'axes.titlesize': 16,
    'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12, 'figure.figsize': (12, 6), 'lines.linewidth': 2,
    'lines.markersize': 6, 'grid.color': 'lightgray', 'grid.linestyle': ':',
    'grid.linewidth': 0.6, 'axes.edgecolor': 'black', 'xtick.color': 'black',
    'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.titlecolor': 'black',
    'text.color': 'black', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.edgecolor': 'white',
    'legend.frameon': True, 'legend.facecolor': 'white', 'legend.edgecolor': 'gray',
})

# --- Constants and Helpers (must match precompute_pdf.py) ---
rho_eq = 1512.0
alpha = 0.1
sigma_eq_val = np.sqrt(0.005)

# Mass Function & Moments (must match precompute_pdf.py)
m1_peak = 1.0; m2_peak = 1.0e-4; sigma_rel = 0.01
def calculate_analytical_moments():
    denom_avM = 0.5/m1_peak + 0.5/m2_peak
    avM = 1.0 / denom_avM if denom_avM > 0 else 1.0
    avM2 = 0.5 * m1_peak**2 + 0.5 * m2_peak**2
    return avM, avM2
AVERAGE_MASS_PSI, _ = calculate_analytical_moments()
print(f"Using analytical <m> = {AVERAGE_MASS_PSI:.3g} M_sun")

# Helper functions for simplified model
def calculate_X_from_a(a, f, M_avg_psi):
    if a <= 0 or f <= 0 or M_avg_psi <= 0: return 0.0
    xb_cubed = 3.0 * M_avg_psi / (4.0 * np.pi * f * rho_eq)
    if xb_cubed <= 0: return 0.0
    xb = xb_cubed**(1.0/3.0)
    term_inside = (a * (f**(1.0/3.0))) / (alpha * xb) if (alpha > 0 and xb > 0) else 0.0
    return term_inside**(0.75) if term_inside >= 0 else 0.0

def calculate_j0(a, f, M1, M2, M_avg_psi, c_j=1.0):
    M_total = M1 + M2
    if M_total <= 0 or M_avg_psi <=0: return 0.0
    X = calculate_X_from_a(a, f, M_avg_psi)
    if X <= 0: return 0.0
    j0 = c_j * X * M_avg_psi / M_total
    return max(j0, 1e-99)

# --- SIMPLIFIED P(a,j) Model (Raidal Eq. 2.32) ---
def calculate_P_la_lj_simplified(la, lj, f_pbh, M1, M2, M_avg_psi, sigma_M):
    """
    Calculates the P(log a, log j) value using the SIMPLIFIED model
    (Raidal Eq. 2.32, Ali-Haimoud Eq. 19)
    """
    a = 10**la; j = 10**lj
    if not (a > 0 and j > 1e-9 and j <= 1.0): return 0.0

    # 1. P(a) factor
    X = calculate_X_from_a(a, f_pbh, M_avg_psi)
    P_a_factor = np.exp(-X) * (a**(-0.25)) if a > 0 else 0.0
    if not (np.isfinite(P_a_factor) and P_a_factor > 0):
        return 0.0

    # 2. j0 and sigma_jM_sq (depend on a)
    j0 = calculate_j0(a, f_pbh, M1, M2, M_avg_psi)
    sigma_jM_sq_term = (3.0/10.0) * (sigma_M**2 / f_pbh**2) * (j0**2)
    
    # Combine j0 and sigma_M contribution (Ali-Haimoud Eq. 22)
    jX_sq = j0**2 + sigma_jM_sq_term
    if jX_sq <= 0: return 0.0
    jX = np.sqrt(jX_sq)

    # 3. P(j|a) factor (Ali-Haimoud Eq. 19)
    gamma = j / jX
    denominator_term = 1.0 + gamma**2
    if denominator_term <= 0: return 0.0
    P_gamma = (gamma**2) / (denominator_term**1.5)
    prob_density_j_given_a = P_gamma / j
    
    if not (np.isfinite(prob_density_j_given_a) and prob_density_j_given_a > 0):
        return 0.0
        
    # 4. Combine and apply Jacobian
    P_a_j_unnorm = prob_density_j_given_a * P_a_factor
    pdf_la_lj = P_a_j_unnorm * a * j * (np.log(10)**2)
    
    return pdf_la_lj if (np.isfinite(pdf_la_lj) and pdf_la_lj > 0) else 0.0
# --- End Simplified Model ---


# --- Main Plotting Function ---
def plot_pdf_comparison(npz_filename):
    
    # 1. Load the precomputed grid
    try:
        data = np.load(npz_filename)
        la_grid = data['la_grid']
        lj_grid = data['lj_grid']
        pdf_grid_precomputed = data['pdf_grid']
        print(f"Loaded precomputed grid from {npz_filename}")
        print(f"  Grid shape: {pdf_grid_precomputed.shape}")
    except Exception as e:
        print(f"ERROR: Could not load file {npz_filename}. {e}")
        return

    # Extract parameters from filename (assumes format: ...M1_X_M2_Y_f_Z.npz)
    try:
        parts = npz_filename.split('_')
        M1_run = float(parts[2])
        M2_run = float(parts[4])
        f_run = float(parts[6].replace('.npz', ''))
        print(f"  Parameters: M1={M1_run:.1e}, M2={M2_run:.1e}, f={f_run:.1e}")
    except Exception:
        print("Warning: Could not parse parameters from filename. Using defaults.")
        M1_run = 1.0
        M2_run = 1e-4
        f_run = 0.01

    # 2. Calculate the simplified analytical PDF on the same grid
    print("Calculating simplified analytical PDF for comparison...")
    N_a, N_j = pdf_grid_precomputed.shape
    pdf_grid_simplified = np.zeros((N_a, N_j))
    
    for i, la in enumerate(la_grid):
        for k, lj in enumerate(lj_grid):
            pdf_grid_simplified[i, k] = calculate_P_la_lj_simplified(
                la, lj, f_run, M1_run, M2_run, AVERAGE_MASS_PSI, sigma_eq_val
            )
            
    # Normalize the simplified grid
    integral_simp = np.trapz(np.trapz(pdf_grid_simplified, lj_grid, axis=1), la_grid, axis=0)
    if integral_simp > 0:
        pdf_grid_simplified /= integral_simp
        print(f"Simplified grid normalized by integral value: {integral_simp:.3g}")
    else:
        print("Warning: Simplified grid integral is zero.")
        
    print("... Simplified PDF calculation complete.")

    # 3. Create the 2-panel plot
    print("Generating plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'P(a, j) Comparison for M1={M1_run:.1e}, M2={M2_run:.1e}, f={f_run:.1e}', fontsize=18)
    
    # Use log(PDF) for better color contrast, add small floor
    pdf_floor = 1e-10
    log_pdf_precomputed = np.log10(pdf_grid_precomputed + pdf_floor)
    log_pdf_simplified = np.log10(pdf_grid_simplified + pdf_floor)
    
    # Determine common color scale
    vmin = max(np.min(log_pdf_precomputed), np.min(log_pdf_simplified))
    vmax = max(np.max(log_pdf_precomputed), np.max(log_pdf_simplified))
    # Adjust vmin if it's too low (e.g., from the floor)
    vmin = max(vmax - 6, vmin) # Show 6 orders of magnitude, or more
    
    levels = np.linspace(vmin, vmax, 10) # 10 contour levels

    # --- Panel 1: Precomputed (Full) PDF ---
    ax1.set_title("Full Model (Precomputed Grid)")
    cf1 = ax1.contourf(10**la_grid, 10**lj_grid, log_pdf_precomputed.T, levels=levels, cmap='viridis')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Semi-major axis, $a$ [pc]')
    ax1.set_ylabel('Ang. Momentum, $j = \\sqrt{1-e^2}$')
    
    # --- Panel 2: Simplified PDF ---
    ax2.set_title("Simplified Model (Raidal Eq. 2.32)")
    cf2 = ax2.contourf(10**la_grid, 10**lj_grid, log_pdf_simplified.T, levels=levels, cmap='viridis')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Semi-major axis, $a$ [pc]')
    ax2.set_ylabel('$j$') # Shorter label
    
    # --- Colorbar ---
    fig.colorbar(cf2, ax=[ax1, ax2], orientation='vertical', label='log10( P(la, lj) ) [Normalized]')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    plot_filename = f"pdf_comparison_{M1_run:.1e}_{M2_run:.1e}_{f_run:.1e}.pdf"
    plt.savefig(plot_filename, dpi=300, format='pdf')
    print(f"\nComparison plot saved to {plot_filename}")
    plt.close(fig)

# --- Main execution ---
if __name__ == "__main__":
    
    # --- Parameters for the grid to LOAD ---
    # (These MUST match the .npz file you want to plot)
    f_pbh_run = 0.01
    M1_run = 1.0
    M2_run = 1e-4
    
    # Construct the filename to load
    filename_to_load = f"pdf_grid_M1_{M1_run:.1e}_M2_{M2_run:.1e}_f_{f_pbh_run:.1e}.npz"

    if not os.path.exists(filename_to_load):
        print(f"ERROR: File '{filename_to_load}' not found.")
        print("Please run precompute_pdf.py first for these parameters.")
    else:
        plot_pdf_comparison(filename_to_load)
