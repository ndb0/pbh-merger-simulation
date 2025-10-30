# compare.py
# Compares analytical and simulation rates for dR/dlnm1/dlnm2
# Uses standard plotting style.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate
import scipy.stats # For Gaussian PDF
import os

# Import the analytical calculation function
try:
    from p3bh_merger_rate_callable import calculate_merger_rate_matrix, sigma_eq
except ImportError:
    print("ERROR: Could not import from 'pbh_merger_rate_callable.py'.")
    exit()

# --- Plotting Style ---
mpl.rcParams.update({
    'font.size': 16, 'font.family': 'sans-serif', 'axes.titlesize': 18,
    'axes.labelsize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 12, 'figure.figsize': (8, 6.5), 'lines.linewidth': 2,
    'lines.markersize': 6, 'grid.color': 'lightgray', 'grid.linestyle': ':',
    'grid.linewidth': 0.6, 'axes.edgecolor': 'black', 'xtick.color': 'black',
    'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.titlecolor': 'black',
    'text.color': 'black', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.edgecolor': 'white',
    'legend.frameon': True, 'legend.facecolor': 'white', 'legend.edgecolor': 'gray',
})

# --- Define Bimodal Mass Function (Normalized Analytically) ---
m1_peak = 1.0; m2_peak = 1.0e-4; sigma_rel = 0.01
sigma_g1 = max(sigma_rel * m1_peak, 1e-12 * m1_peak)
sigma_g2 = max(sigma_rel * m2_peak, 1e-12 * m2_peak)

def mass_func_psi_normalized(m):
    """Normalized bimodal Gaussian mass function psi(m), integral dm = 1."""
    pdf1 = 0.5 * scipy.stats.norm.pdf(m, loc=m1_peak, scale=sigma_g1)
    pdf2 = 0.5 * scipy.stats.norm.pdf(m, loc=m2_peak, scale=sigma_g2)
    result = pdf1 + pdf2
    return np.maximum(0.0, result) if isinstance(m, np.ndarray) else max(0.0, result)

# Create interpolator for the correctly normalized psi(m)
m_min_psi = m2_peak / 1000; m_max_psi = m1_peak * 10
num_points_psi = 1000
mass_grid_psi = np.logspace(np.log10(m_min_psi), np.log10(m_max_psi), num_points_psi)
psi_vals_grid = mass_func_psi_normalized(mass_grid_psi)
norm_check = scipy.integrate.trapz(psi_vals_grid, mass_grid_psi)
print(f"Numerical check of psi(m) normalization (trapz): {norm_check:.4g}")
unique_m_psi, unique_idx_psi = np.unique(mass_grid_psi, return_index=True)
if len(unique_m_psi) < 2: raise ValueError("Need >= 2 unique masses for psi interpolation.")
try:
    psi_interp_callable = scipy.interpolate.interp1d(
        unique_m_psi, psi_vals_grid[unique_idx_psi],
        bounds_error=False, fill_value=0.0, kind='linear'
    )
except ValueError as e:
    print(f"Error creating psi interpolator: {e}"); psi_interp_callable = lambda m: 0.0

# --- Define f_PBH Range and Mass Pairs ---
f_pbh_values_plot = np.logspace(-4, 0, 50)
mass_pairs_sim = [(1.0, 1.0), (1e-4, 1e-4), (1.0, 1e-4)]
colors = {'1.0-1.0': '#e41a1c', '1.0e-04-1.0e-04': '#377eb8', '1.0-1.0e-04': '#4daf4a'}
labels = {'1.0-1.0': '$M_1$-$M_1$ (1 $M_\\odot$)',
          '1.0e-04-1.0e-04': '$M_2$-$M_2$ (10$^{-4} M_\\odot$)',
          '1.0-1.0e-04': '$M_1$-$M_2$'}

# --- Calculate Analytical Rates ---
print("\nCalculating Analytical Rates (VERBOSE)...")
analytical_total_rates = []
analytical_pair_rates = {f"{m1:.1e}-{m2:.1e}": [] for m1, m2 in mass_pairs_sim}
analytical_mass_grid = np.logspace(np.log10(m_min_psi), np.log10(m_max_psi), 200) # Increased grid

for f_val in f_pbh_values_plot:
    print(f"  f = {f_val:.3e}:")
    try:
        m_out, rate_mat, _, total_rate = calculate_merger_rate_matrix(
            f_total=f_val,
            psi_interp_func=psi_interp_callable,
            mlist=analytical_mass_grid,
            m1_peak=m1_peak, m2_peak=m2_peak, use_analytical_moments=True
        )
        if not np.isfinite(total_rate) or abs(total_rate) > 1e20: total_rate = np.nan
        analytical_total_rates.append(total_rate)
        print(f"    Total Rate = {total_rate:.3g}")

        # print("    Pair Rates (dR/dlnm1 dlnm2):") # Reduce print clutter
        print("    Pair Rates (dR/dlnm1 dlnm2):")
        for m1_sim, m2_sim in mass_pairs_sim:
                idx1 = np.argmin(np.abs(analytical_mass_grid - m1_sim))
                idx2 = np.argmin(np.abs(analytical_mass_grid - m2_sim))
                # Correct line:
                pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}" # Use m2_sim here
                rate_val = rate_mat[idx1, idx2]
                if not np.isfinite(rate_val) or abs(rate_val) > 1e20: rate_val = np.nan
                analytical_pair_rates[pair_key].append(rate_val)
                print(f"      {pair_key}: {rate_val:.3g}")

    except Exception as e:
        print(f"    ERROR calculating analytical rate for f={f_val:.3e}: {e}")
        import traceback; traceback.print_exc()
        analytical_total_rates.append(np.nan)
        for m1_sim, m2_sim in mass_pairs_sim:
             analytical_pair_rates[f"{m1_sim:.1e}-{m2_sim:.1e}"].append(np.nan)

print("... Analytical Rates Calculation Finished.")
analytical_total_rates = np.array(analytical_total_rates)
for key in analytical_pair_rates:
    analytical_pair_rates[key] = np.array(analytical_pair_rates[key])
if np.all(np.isnan(analytical_total_rates)): print("\nERROR: All analytical total rates NaN.")
elif np.nanmax(analytical_total_rates) > 1e10: print("\nWARNING: Analytical rates seem large.")

# --- Load Simulation Results ---
print("\nLoading Simulation Results and Estimating Rates (VERBOSE)...")
simulation_rates = {f"{m1:.1e}-{m2:.1e}": [] for m1, m2 in mass_pairs_sim}
simulation_f_values = {f"{m1:.1e}-{m2:.1e}": [] for m1, m2 in mass_pairs_sim}
sim_output_dir = "simulation_output"
f_pbh_simulated = np.logspace(-4., -1., 5) # Match f values from unequal.py

for m1_sim, m2_sim in mass_pairs_sim:
    pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"; print(f"  Pair {pair_key}:") # Use m2_sim
    for f_val in f_pbh_simulated:
        f_key = f"f_{f_val:.3e}"
        sim_file = os.path.join(sim_output_dir, f"results_M1_{m1_sim:.1e}_M2_{m2_sim:.1e}_{f_key}.npz")
        if os.path.exists(sim_file):
            try:
                data = np.load(sim_file)
                frac_initial = data['frac_initial']; frac_remapped = data['frac_remapped']
                print(f"    f = {f_val:.3e}: F_init={frac_initial:.3g}, F_remap={frac_remapped:.3g}")

                log_f_analytical = np.log10(f_pbh_values_plot)
                an_rates_pair = analytical_pair_rates[pair_key]
                valid_an_mask = np.isfinite(an_rates_pair) & (an_rates_pair > 0)
                if np.sum(valid_an_mask) >= 2:
                    log_rate_analytical_pair = np.log10(an_rates_pair[valid_an_mask])
                    # Use numpy interpolation
                    log_rate_an_interp = np.interp(np.log10(f_val),
                                                   log_f_analytical[valid_an_mask],
                                                   log_rate_analytical_pair)
                    rate_an_interp = 10**log_rate_an_interp
                else: rate_an_interp = 0.0
                # print(f"      Analytical Rate Density @ f={f_val:.3e}: {rate_an_interp:.3g}") # Reduce clutter

                if frac_initial > 1e-9: sim_rate_est = rate_an_interp * (frac_remapped / frac_initial)
                elif frac_remapped > 1e-9: sim_rate_est = rate_an_interp * frac_remapped / 1e-6
                else: sim_rate_est = 0.0
                if not np.isfinite(sim_rate_est) or sim_rate_est < 0: sim_rate_est = 0.0

                simulation_f_values[pair_key].append(f_val)
                simulation_rates[pair_key].append(sim_rate_est)
                print(f"      Estimated Simulation Rate = {sim_rate_est:.3g}")
            except Exception as e: print(f"    ERROR loading/processing {sim_file}: {e}")
        else: print(f"    f = {f_val:.3e}: Simulation result file not found.")
    simulation_f_values[pair_key] = np.array(simulation_f_values[pair_key])
    simulation_rates[pair_key] = np.array(simulation_rates[pair_key])
print("... Simulation Rates Estimation Finished.")

# --- Create Plot ---
print("\nGenerating Plot...")
fig, ax = plt.subplots(figsize=(8, 6.5))

# Total Analytical Rate
valid_total = np.isfinite(analytical_total_rates) & (analytical_total_rates > 0)
if np.any(valid_total): ax.loglog(f_pbh_values_plot[valid_total], analytical_total_rates[valid_total], color='black', linestyle='-', linewidth=2.5, label='Total Analytical Rate (Integrated)', zorder=10) # Thicker, on top
else: print("Warning: No valid total analytical rates.")

# Pair Rates
plot_sim_success = False
for m1_sim, m2_sim in mass_pairs_sim:
    pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"; color = colors.get(pair_key, 'dimgray'); label = labels.get(pair_key, pair_key) # Use m2_sim
    # Analytical Density
    an_rates = analytical_pair_rates[pair_key]; valid_an = np.isfinite(an_rates) & (an_rates > 0)
    if np.any(valid_an): ax.loglog(f_pbh_values_plot[valid_an], an_rates[valid_an], color=color, linestyle='--', linewidth=1.5, label=f'{label} (Analytical Density)')
    else: print(f"Warning: No valid analytical rates for {pair_key}")
    # Simulation Rate
    sim_fs = simulation_f_values[pair_key]; sim_rates = simulation_rates[pair_key]; valid_sim = np.isfinite(sim_rates) & (sim_rates > 0)
    if np.any(valid_sim):
        ax.loglog(sim_fs[valid_sim], sim_rates[valid_sim], color=color, linestyle='-', marker='o', markersize=6, linewidth=2.0, label=f'{label} (Simulation w/ Remapping)')
        plot_sim_success = True
    else: print(f"Warning: No valid simulation rates for {pair_key}")
if not plot_sim_success: print("\nWARNING: NO VALID SIMULATION DATA PLOTTED.")

# LVK Band
lvk_min = 10; lvk_max = 200
ax.fill_between(f_pbh_values_plot, lvk_min, lvk_max, color='gray', alpha=0.4, label='LVK O3 Rate Range (Approx)', zorder=1) # Standard gray, behind lines

# --- Customize Plot ---
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1.e-5, 1.); ax.set_ylim(1e-6, 1e6)
ax.set_xlabel(r'$f_{\rm PBH}$'); ax.set_ylabel(r'Merger Rate [Gpc$^{-3}$ yr$^{-1}$]')
ax.set_title('PBH Merger Rate: Analytical vs. Simulation (Unequal Mass)')
ax.legend(loc='lower right', fontsize=11) # Adjusted font size
ax.grid(True, which='both', linestyle=':', linewidth=0.6, color='lightgray')
fig.tight_layout()

plot_filename = "pbh_merger_rate_comparison_unequal_whiteBG.pdf"
plt.savefig(plot_filename, dpi=300); print(f"\nComparison plot saved to {plot_filename}")
print("\nPlotting Script Finished.")
