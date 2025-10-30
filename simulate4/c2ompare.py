# plot_comparison.py
# ADDED: Verbosity to print rates

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate
import os

# Import the analytical calculation function
try:
    from pbh_merger_rate_callable import calculate_merger_rate_matrix, sigma_eq # Import sigma_eq too
except ImportError:
    print("ERROR: Could not import 'calculate_merger_rate_matrix' from 'pbh_merger_rate_callable.py'.")
    print("Ensure the file exists and is correctly named.")
    exit()
except NameError:
    print("ERROR: 'sigma_eq' might not be defined globally in 'pbh_merger_rate_callable.py'. Add it.")
    exit()


# --- Plotting Style ---
mpl.rcParams.update({'font.size': 18,'font.family':'sans-serif'})
plt.style.use('dark_background')

# --- Define Bimodal Mass Function (same as used in unequal.py) ---
m1_peak = 1.0; m2_peak = 1.0e-4; sigma_rel = 0.01
sigma_g1 = sigma_rel * m1_peak; sigma_g2 = sigma_rel * m2_peak

def gaussian_sum_unnorm(m, m1, sig1, m2, sig2):
    sig1 = max(sig1, 1e-10 * m1); sig2 = max(sig2, 1e-10 * m2)
    norm1 = 1.0 / (np.sqrt(2 * np.pi) * sig1); norm2 = 1.0 / (np.sqrt(2 * np.pi) * sig2)
    g1 = norm1 * np.exp(-0.5 * ((m - m1) / sig1)**2)
    g2 = norm2 * np.exp(-0.5 * ((m - m2) / sig2)**2)
    return 0.5 * g1 + 0.5 * g2

m_min_psi = m2_peak / 100; m_max_psi = m1_peak * 10
num_points_psi = 500
mass_grid_psi = np.logspace(np.log10(m_min_psi), np.log10(m_max_psi), num_points_psi)

# Normalize psi(m) such that integral(psi(m) dm) = 1
norm_constant_psi, norm_err = scipy.integrate.quad(
    lambda m: gaussian_sum_unnorm(m, m1_peak, sigma_g1, m2_peak, sigma_g2),
    mass_grid_psi[0], mass_grid_psi[-1], limit=200, epsabs=1e-9
)
print(f"Normalization constant for psi(m): {norm_constant_psi:.4g} (Error estimate: {norm_err:.2g})")
if norm_constant_psi <= 0: raise ValueError("Psi normalization failed.")

def mass_func_psi_normalized(m):
    # Ensure function handles arrays from quad/trapz if needed, though called point-wise here
    m = np.asarray(m)
    result = np.zeros_like(m, dtype=float)
    valid_mask = (m > 0)
    result[valid_mask] = gaussian_sum_unnorm(m[valid_mask], m1_peak, sigma_g1, m2_peak, sigma_g2) / norm_constant_psi
    return result[0] if np.isscalar(m) else result # Return scalar if input was scalar


# --- Define f_PBH Range and Mass Pairs ---
f_pbh_values_plot = np.logspace(-4, 0, 50)
mass_pairs_sim = [(1.0, 1.0), (1e-4, 1e-4), (1.0, 1e-4)]
colors = {'1.0-1.0': 'red', '1.0e-04-1.0e-04': 'blue', '1.0-1.0e-04': 'green'}
labels = {'1.0-1.0': '$M_1$-$M_1$ (1 $M_\\odot$)',
          '1.0e-04-1.0e-04': '$M_2$-$M_2$ (10$^{-4} M_\\odot$)',
          '1.0-1.0e-04': '$M_1$-$M_2$'}

# --- Calculate Analytical Rates ---
print("\nCalculating Analytical Rates (VERBOSE)...")
analytical_total_rates = []
analytical_pair_rates = {f"{m1:.1e}-{m2:.1e}": [] for m1, m2 in mass_pairs_sim}
# Use a finer grid for analytical calculation stability
analytical_mass_grid = np.logspace(np.log10(m_min_psi), np.log10(m_max_psi), 200)

for f_val in f_pbh_values_plot:
    print(f"  f = {f_val:.3e}:")
    try:
        m_out, rate_mat, fpbh0_calc, total_rate = calculate_merger_rate_matrix(
            f_total=f_val,
            mass_func_psi=mass_func_psi_normalized,
            mlist=analytical_mass_grid
        )
        analytical_total_rates.append(total_rate)
        print(f"    Total Rate = {total_rate:.3g}")

        # Extract rates for specific pairs
        print("    Pair Rates (dR/dlnm1 dlnm2):")
        for m1_sim, m2_sim in mass_pairs_sim:
            idx1 = np.argmin(np.abs(analytical_mass_grid - m1_sim))
            idx2 = np.argmin(np.abs(analytical_mass_grid - m2_sim))
            pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"
            rate_val = rate_mat[idx1, idx2]
            analytical_pair_rates[pair_key].append(rate_val)
            print(f"      {pair_key}: {rate_val:.3g}")

    except Exception as e:
        print(f"    ERROR calculating analytical rate for f={f_val:.3e}: {e}")
        analytical_total_rates.append(np.nan) # Append NaN on error
        for m1_sim, m2_sim in mass_pairs_sim:
             pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"
             analytical_pair_rates[pair_key].append(np.nan)


print("... Analytical Rates Calculation Finished.")

analytical_total_rates = np.array(analytical_total_rates)
for key in analytical_pair_rates:
    analytical_pair_rates[key] = np.array(analytical_pair_rates[key])

# Check for NaN values which would blank the plot
if np.all(np.isnan(analytical_total_rates)):
    print("\nERROR: All calculated analytical total rates are NaN. Check calculation logic.")
# Check magnitudes again
if np.nanmax(analytical_total_rates) > 1e15: # Arbitrary large number check
    print("\nWARNING: Analytical rates seem excessively large. Check units/constants in pbh_merger_rate_callable.py")

# --- Load Simulation Results and Calculate Simulation Rates ---
print("\nLoading Simulation Results and Estimating Rates (VERBOSE)...")
simulation_rates = {f"{m1:.1e}-{m2:.1e}": [] for m1, m2 in mass_pairs_sim}
simulation_f_values = {f"{m1:.1e}-{m2:.1e}": [] for m1, m2 in mass_pairs_sim} # Store f values per pair

sim_output_dir = "simulation_output"
f_pbh_simulated = np.logspace(-4., -1., 5) # Match f values from unequal.py

for m1_sim, m2_sim in mass_pairs_sim:
    pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"
    print(f"  Pair {pair_key}:")

    for f_val in f_pbh_simulated:
        f_key = f"f_{f_val:.3e}"
        sim_file = os.path.join(sim_output_dir, f"results_M1_{m1_sim:.1e}_M2_{m2_sim:.1e}_{f_key}.npz")

        if os.path.exists(sim_file):
            try:
                data = np.load(sim_file)
                frac_initial = data['frac_initial']
                frac_remapped = data['frac_remapped']
                print(f"    f = {f_val:.3e}: F_init={frac_initial:.3g}, F_remap={frac_remapped:.3g}")

                # Find corresponding analytical rate density for this f_val
                # Interpolate log rates in log space for better accuracy
                log_f_analytical = np.log10(f_pbh_values_plot)
                log_rate_analytical_pair = np.log10(analytical_pair_rates[pair_key])
                # Filter out NaNs before interpolation
                valid_an_rate = np.isfinite(log_rate_analytical_pair)
                if np.any(valid_an_rate) and len(log_f_analytical[valid_an_rate]) > 1:
                     log_rate_an_interp = np.interp(np.log10(f_val), log_f_analytical[valid_an_rate], log_rate_analytical_pair[valid_an_rate])
                     rate_an_interp = 10**log_rate_an_interp
                else:
                     print(f"      Warning: Could not interpolate analytical rate for f={f_val:.3e}. Using fallback.")
                     # Fallback: find closest value
                     idx_f_an = np.argmin(np.abs(f_pbh_values_plot - f_val))
                     rate_an_interp = analytical_pair_rates[pair_key][idx_f_an]
                     if not np.isfinite(rate_an_interp): rate_an_interp = 0.0 # Use 0 if fallback is NaN

                print(f"      Analytical Rate Density @ f={f_val:.3e}: {rate_an_interp:.3g}")

                # Calculate simulation rate
                if frac_initial > 1e-9:
                    rate_ratio = frac_remapped / frac_initial
                    sim_rate_est = rate_an_interp * rate_ratio
                elif frac_remapped > 1e-9:
                    sim_rate_est = rate_an_interp * frac_remapped / 1e-6 # Arbitrary scaling
                    # print(f"      Warning: F_initial near zero. Sim rate estimate might be inaccurate.")
                else: sim_rate_est = 0.0

                simulation_f_values[pair_key].append(f_val)
                simulation_rates[pair_key].append(sim_rate_est)
                print(f"      Estimated Simulation Rate = {sim_rate_est:.3g}")

            except Exception as e:
                print(f"    ERROR loading/processing {sim_file}: {e}")
        else:
            print(f"    f = {f_val:.3e}: Simulation result file not found.")

    simulation_f_values[pair_key] = np.array(simulation_f_values[pair_key])
    simulation_rates[pair_key] = np.array(simulation_rates[pair_key])

print("... Simulation Rates Estimation Finished.")

# --- Create Plot ---
print("\nGenerating Plot...")
plt.figure(figsize=(10, 8))

# Plot Total Analytical Rate (only if finite values exist)
valid_total_an = np.isfinite(analytical_total_rates) & (analytical_total_rates > 0)
if np.any(valid_total_an):
    plt.loglog(f_pbh_values_plot[valid_total_an], analytical_total_rates[valid_total_an], color='white', linestyle='-', linewidth=2, label='Total Analytical Rate (Integrated)')
else:
    print("Warning: No valid total analytical rates to plot.")

# Plot Analytical and Simulation Pair Rate Densities
plot_sim_success = False
for m1_sim, m2_sim in mass_pairs_sim:
    pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"
    color = colors.get(pair_key, 'gray')
    label = labels.get(pair_key, pair_key)

    # Analytical Density (plot only finite, positive values)
    an_rates = analytical_pair_rates[pair_key]
    valid_an = np.isfinite(an_rates) & (an_rates > 0)
    if np.any(valid_an):
        plt.loglog(f_pbh_values_plot[valid_an], an_rates[valid_an], color=color, linestyle='--', linewidth=1.5, label=f'{label} (Analytical Density)')
    else:
         print(f"Warning: No valid analytical rates to plot for pair {pair_key}")


    # Simulation Rate (plot only finite, positive values)
    sim_fs = simulation_f_values[pair_key]
    sim_rates = simulation_rates[pair_key]
    valid_sim = np.isfinite(sim_rates) & (sim_rates > 0)
    if np.any(valid_sim):
        plt.loglog(sim_fs[valid_sim], sim_rates[valid_sim], color=color, linestyle='-', marker='o', markersize=5, linewidth=1.5, label=f'{label} (Simulation w/ Remapping)')
        plot_sim_success = True # Mark that we plotted at least some sim data
    else:
        print(f"Warning: No valid simulation rates to plot for pair {pair_key}")

if not plot_sim_success:
     print("\nWARNING: NO VALID SIMULATION DATA WAS PLOTTED.")

# Add LVK Bound Region
lvk_min = 10; lvk_max = 200
plt.fill_between(f_pbh_values_plot, lvk_min, lvk_max, color='gray', alpha=0.3, label='LVK O3 Rate Range (Approx)')

# --- Customize Plot ---
plt.xlim(1.e-4, 1.)
# Adjust ylim based on expected physical rates, e.g., 1e-2 to 1e6
#plt.ylim(1e-2, 1e6)
plt.xlabel(r'$f_{\rm PBH}$')
plt.ylabel(r'Merger Rate [Gpc$^{-3}$ yr$^{-1}$]')
plt.title('PBH Merger Rate: Analytical vs. Simulation (Unequal Mass)')
plt.legend(loc='lower right', fontsize='small')
plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray')
plt.tight_layout()

# Save plot
plot_filename = "pbh_merger_rate_comparison_unequal.png"
plt.savefig(plot_filename)
print(f"\nComparison plot saved to {plot_filename}")
# plt.show()

print("\nPlotting Script Finished.")
