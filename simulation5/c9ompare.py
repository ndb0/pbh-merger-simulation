# compare.py
# Compares analytical and simulation rates for dR/dlnm1/dlnm2
# REVISED: Restored print statements for pair rates, standard plot style.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate
import scipy.stats # For Gaussian PDF
import os
import time # For timing

# Import the analytical calculation function
try:
    from p9bh_merger_rate_callable import calculate_merger_rate_matrix, sigma_eq
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
    pdf1 = 0.5 * scipy.stats.norm.pdf(m, loc=m1_peak, scale=sigma_g1)
    pdf2 = 0.5 * scipy.stats.norm.pdf(m, loc=m2_peak, scale=sigma_g2)
    result = pdf1 + pdf2
    return np.maximum(0.0, result) if isinstance(m, np.ndarray) else max(0.0, result)

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

# --- Plot the Mass Function ---
# ... (Keep the psi plot generation code from the previous response) ...
print("\nGenerating Mass Function Plot...")
fig_psi, ax_psi = plt.subplots(figsize=(8, 6))
ax_psi.loglog(mass_grid_psi, psi_vals_grid, color='blue', linewidth=2)
ax_psi.axvline(m1_peak, color='red', linestyle='--', alpha=0.7, label=f'$M_1$ Peak ({m1_peak}$M_\\odot$)')
ax_psi.axvline(m2_peak, color='purple', linestyle='--', alpha=0.7, label=f'$M_2$ Peak ({m2_peak}$M_\\odot$)')
ax_psi.set_xlim(m_min_psi, m_max_psi)
min_psi_val = np.min(psi_vals_grid[psi_vals_grid > 1e-99]) if np.any(psi_vals_grid > 1e-99) else 1e-9
max_psi_val = np.max(psi_vals_grid); ymin_psi = max(1e-9, min_psi_val / 10); ymax_psi = max_psi_val * 10
if ymax_psi <= ymin_psi: ymax_psi = ymin_psi * 1e6
ax_psi.set_ylim(ymin_psi, ymax_psi)
ax_psi.set_xlabel('PBH Mass $m$ [$M_\\odot$]'); ax_psi.set_ylabel('Mass Function $\\psi(m)$ [$M_\\odot^{-1}$]')
ax_psi.set_title('Input PBH Mass Function $\\psi(m)$'); ax_psi.legend(fontsize='small')
ax_psi.grid(True, which='both', linestyle=':', linewidth=0.6, color='lightgray'); fig_psi.tight_layout()
psi_plot_filename = "psi_mass_function.png"
plt.savefig(psi_plot_filename, dpi=300); print(f"Mass function plot saved to {psi_plot_filename}"); plt.close(fig_psi)
# --- End Mass Function Plot ---

# --- Define f_PBH Range and Mass Pairs ---
f_pbh_values_plot = np.logspace(-4, 0, 50)
mass_pairs_sim = [(1.0, 1.0), (1e-4, 1e-4), (1.0, 1e-4)]
colors = {'1.0e+00-1.0e+00': 'red', '1.0e-04-1.0e-04': 'blue', '1.0e+00-1.0e-04': 'green'}
labels = {'1.0e+00-1.0e+00': '$M_1$-$M_1$ (1 $M_\\odot$)',
          '1.0e-04-1.0e-04': '$M_2$-$M_2$ (10$^{-4} M_\\odot$)',
          '1.0e+00-1.0e-04': '$M_1$-$M_2$'}

# --- Calculate Analytical Rates ---
print("\nCalculating Analytical Rates (VERBOSE)...")
analytical_total_rates = []
analytical_pair_rates = {f"{m1:.1e}-{m2:.1e}": [] for m1, m2 in mass_pairs_sim}
# === REDUCE GRID SIZE FOR SPEED ===
analytical_mass_grid = np.logspace(np.log10(m_min_psi), np.log10(m_max_psi), 150) # Reduced from 500
# ==================================

analytical_start_time = time.time() # Start timing

for f_val in f_pbh_values_plot:
    print(f"  f = {f_val:.3e}:")
    f_start_time = time.time() # Time each f value
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

        # === RESTORED PRINT BLOCK ===
        print("    Pair Rates (dR/dlnm1 dlnm2):")
        for m1_sim, m2_sim in mass_pairs_sim:
            idx1 = np.argmin(np.abs(analytical_mass_grid - m1_sim))
            idx2 = np.argmin(np.abs(analytical_mass_grid - m2_sim))
            pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"
            rate_val = rate_mat[idx1, idx2]
            if not np.isfinite(rate_val) or abs(rate_val) > 1e20: rate_val = np.nan
            if pair_key in analytical_pair_rates:
                analytical_pair_rates[pair_key].append(rate_val)
            print(f"      {pair_key}: {rate_val:.3g}") # Print the specific pair rate
        # ============================

    except Exception as e:
        print(f"    ERROR calculating analytical rate for f={f_val:.3e}: {e}")
        import traceback; traceback.print_exc()
        analytical_total_rates.append(np.nan)
        for m1_sim, m2_sim in mass_pairs_sim:
             analytical_pair_rates[f"{m1_sim:.1e}-{m2_sim:.1e}"].append(np.nan)

    f_end_time = time.time()
    print(f"    Time for this f: {f_end_time - f_start_time:.2f} s") # Print time per f

analytical_end_time = time.time()
print(f"... Analytical Rates Calculation Finished ({analytical_end_time - analytical_start_time:.2f} s total).")

# ... (Rest of the script: Convert to arrays, Load Sim Results, Plotting - remains the same) ...
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
f_pbh_simulated = np.logspace(-4., -1., 5)

for m1_sim, m2_sim in mass_pairs_sim:
    pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"; print(f"  Pair {pair_key}:")
    for f_val in f_pbh_simulated:
        f_key = f"f_{f_val:.3e}"
        sim_file_pair_key = f"M1_{m1_sim:.1e}_M2_{m2_sim:.1e}"
        sim_file = os.path.join(sim_output_dir, f"results_{sim_file_pair_key}_{f_key}.npz")
        if os.path.exists(sim_file):
            try:
                data = np.load(sim_file)
                frac_initial = data['frac_initial']; frac_remapped = data['frac_remapped']
                print(f"    f = {f_val:.3e}: F_init={frac_initial:.3g}, F_remap={frac_remapped:.3g}")

                log_f_analytical = np.log10(f_pbh_values_plot)
                if pair_key in analytical_pair_rates:
                    an_rates_pair = analytical_pair_rates[pair_key]
                else: print(f"      Error: Key '{pair_key}' not found."); continue
                valid_an_mask = np.isfinite(an_rates_pair) & (an_rates_pair > 0)
                if np.sum(valid_an_mask) >= 2:
                    log_rate_analytical_pair = np.log10(an_rates_pair[valid_an_mask])
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
if np.any(valid_total): ax.loglog(f_pbh_values_plot[valid_total], analytical_total_rates[valid_total], color='black', linestyle='-', linewidth=2.5, label='Total Analytical Rate (Integrated)', zorder=10)
else: print("Warning: No valid total analytical rates.")

# Pair Rates
plot_sim_success = False
for m1_sim, m2_sim in mass_pairs_sim:
    pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"; color = colors.get(pair_key, 'gray'); label = labels.get(pair_key, pair_key)
    # Analytical Density
    if pair_key in analytical_pair_rates:
        an_rates = analytical_pair_rates[pair_key]; valid_an = np.isfinite(an_rates) & (an_rates > 0)
        if np.any(valid_an): ax.loglog(f_pbh_values_plot[valid_an], an_rates[valid_an], color=color, linestyle='--', linewidth=1.5, label=f'{label} (Analytical Density)')
        else: print(f"Warning: No valid analytical rates for {pair_key}")
    else: print(f"Warning: Key '{pair_key}' missing for analytical rates plot.")
    # Simulation Rate
    if pair_key in simulation_f_values and pair_key in simulation_rates:
        sim_fs = simulation_f_values[pair_key]; sim_rates = simulation_rates[pair_key]; valid_sim = np.isfinite(sim_rates) & (sim_rates > 0)
        if np.any(valid_sim):
            ax.loglog(sim_fs[valid_sim], sim_rates[valid_sim], color=color, linestyle='-', marker='o', markersize=6, linewidth=2.0, label=f'{label} (Simulation w/ Remapping)')
            plot_sim_success = True
        else: print(f"Warning: No valid simulation rates for {pair_key}")
    else: print(f"Warning: Key '{pair_key}' missing for simulation rates plot.")
if not plot_sim_success: print("\nWARNING: NO VALID SIMULATION DATA PLOTTED.")

# LVK Band
lvk_min = 10; lvk_max = 200
ax.fill_between(f_pbh_values_plot, lvk_min, lvk_max, color='lightgray', alpha=0.6, label='LVK O3 Rate Range (Approx)', zorder=1)

# --- Customize Plot ---
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1.e-4, 1.); ax.set_ylim(1e-6, 1e6)
ax.set_xlabel(r'$f_{\rm PBH}$'); ax.set_ylabel(r'Merger Rate [Gpc$^{-3}$ yr$^{-1}$]')
ax.set_title('PBH Merger Rate: Analytical vs. Simulation (Unequal Mass)')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, which='both', linestyle=':', linewidth=0.6, color='lightgray')
fig.tight_layout()

plot_filename_pdf = "pbh_merger_rate_comparison_unequal_whiteBG.pdf"
plt.savefig(plot_filename_pdf, dpi=300, format='pdf')
print(f"\nComparison plot saved to {plot_filename_pdf}")
print("\nPlotting Script Finished.")
plt.close(fig)
