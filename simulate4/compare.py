# plot_comparison.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate
import os

# Import the analytical calculation function
from pbh_merger_rate_callable import calculate_merger_rate_matrix

# --- Plotting Style ---
mpl.rcParams.update({'font.size': 18,'font.family':'sans-serif'})
# Add other mpl params as needed...
plt.style.use('dark_background') # Match example image style

# --- Define Bimodal Mass Function (same as used in unequal.py) ---
m1_peak = 1.0      # Solar masses
m2_peak = 1.0e-4   # Solar masses
sigma_rel = 0.01   # Relative width
# f_total is varied below

sigma_g1 = sigma_rel * m1_peak
sigma_g2 = sigma_rel * m2_peak

def gaussian_sum_unnorm(m, m1, sig1, m2, sig2):
    sig1 = max(sig1, 1e-10 * m1); sig2 = max(sig2, 1e-10 * m2)
    norm1 = 1.0 / (np.sqrt(2 * np.pi) * sig1); norm2 = 1.0 / (np.sqrt(2 * np.pi) * sig2)
    g1 = norm1 * np.exp(-0.5 * ((m - m1) / sig1)**2)
    g2 = norm2 * np.exp(-0.5 * ((m - m2) / sig2)**2)
    return 0.5 * g1 + 0.5 * g2 # Equal contribution

# Create a mass grid for normalization and integration
m_min_psi = m2_peak / 100
m_max_psi = m1_peak * 10
num_points_psi = 500
mass_grid_psi = np.logspace(np.log10(m_min_psi), np.log10(m_max_psi), num_points_psi)

# Normalize psi(m) such that integral(psi(m) dm) = 1
norm_constant_psi, _ = scipy.integrate.quad(
    lambda m: gaussian_sum_unnorm(m, m1_peak, sigma_g1, m2_peak, sigma_g2),
    mass_grid_psi[0], mass_grid_psi[-1], limit=200
)
if norm_constant_psi <= 0: raise ValueError("Psi normalization failed.")

def mass_func_psi_normalized(m):
    return gaussian_sum_unnorm(m, m1_peak, sigma_g1, m2_peak, sigma_g2) / norm_constant_psi

# --- Define f_PBH Range and Mass Pairs ---
f_pbh_values_plot = np.logspace(-4, 0, 50) # Finer grid for plotting analytical curve
mass_pairs_sim = [(1.0, 1.0), (1e-4, 1e-4), (1.0, 1e-4)] # Pairs simulated
colors = {'1.0-1.0': 'red', '1.0e-04-1.0e-04': 'blue', '1.0-1.0e-04': 'green'}
labels = {'1.0-1.0': '$M_1$-$M_1$ (1 $M_\\odot$)',
          '1.0e-04-1.0e-04': '$M_2$-$M_2$ (10$^{-4} M_\\odot$)',
          '1.0-1.0e-04': '$M_1$-$M_2$'}

# --- Calculate Analytical Rates ---
print("Calculating Analytical Rates...")
analytical_total_rates = []
analytical_pair_rates = {f"{m1:.1e}-{m2:.1e}": [] for m1, m2 in mass_pairs_sim}
analytical_mass_grid = np.logspace(np.log10(m_min_psi), np.log10(m_max_psi), 100) # Grid for rate matrix

for f_val in f_pbh_values_plot:
    m_out, rate_mat, _, total_rate = calculate_merger_rate_matrix(
        f_total=f_val,
        mass_func_psi=mass_func_psi_normalized,
        mlist=analytical_mass_grid
    )
    analytical_total_rates.append(total_rate)

    # Extract rates for specific pairs (find closest grid points)
    for m1_sim, m2_sim in mass_pairs_sim:
        idx1 = np.argmin(np.abs(analytical_mass_grid - m1_sim))
        idx2 = np.argmin(np.abs(analytical_mass_grid - m2_sim))
        pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"
        # Rate matrix is dR/dlnm1 dlnm2
        analytical_pair_rates[pair_key].append(rate_mat[idx1, idx2])

print("... Analytical Rates Calculated.")

# Convert lists to arrays
analytical_total_rates = np.array(analytical_total_rates)
for key in analytical_pair_rates:
    analytical_pair_rates[key] = np.array(analytical_pair_rates[key])


# --- Load Simulation Results and Calculate Simulation Rates ---
print("Loading Simulation Results and Estimating Rates...")
simulation_rates = {f"{m1:.1e}-{m2:.1e}": [] for m1, m2 in mass_pairs_sim}
simulation_f_values = {} # Store f values for which we have simulation data

sim_output_dir = "simulation_output" # Directory where unequal.py saved results

for m1_sim, m2_sim in mass_pairs_sim:
    pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"
    pair_sim_f = []
    pair_sim_rate = []

    # Use f_pbh values that were actually simulated by unequal.py
    f_pbh_simulated = np.logspace(-4., -1., 5) # Match f values from unequal.py

    for f_val in f_pbh_simulated:
        f_key = f"f_{f_val:.3e}"
        sim_file = os.path.join(sim_output_dir, f"results_M1_{m1_sim:.1e}_M2_{m2_sim:.1e}_{f_key}.npz")

        if os.path.exists(sim_file):
            try:
                data = np.load(sim_file)
                frac_initial = data['frac_initial']
                frac_remapped = data['frac_remapped']

                # Find corresponding analytical rate density for this f_val
                # Interpolate the analytical pair rates calculated earlier
                if len(analytical_pair_rates[pair_key]) == len(f_pbh_values_plot):
                     rate_an_interp = np.interp(f_val, f_pbh_values_plot, analytical_pair_rates[pair_key])
                else:
                     print(f"Warning: mismatch in lengths for interpolation f={f_val}")
                     # Find closest analytical f value as fallback
                     idx_f_an = np.argmin(np.abs(f_pbh_values_plot - f_val))
                     rate_an_interp = analytical_pair_rates[pair_key][idx_f_an]


                # Calculate simulation rate: R_an * (F_remap / F_init)
                # Handle F_init = 0 case: if initial merges, but remapped doesn't, rate -> 0
                # If initial doesn't merge, but remapped does, rate -> large? Or use different scaling?
                # Let's cap the ratio or handle zero division.
                if frac_initial > 1e-9: # Avoid division by zero/small numbers
                    rate_ratio = frac_remapped / frac_initial
                    sim_rate_est = rate_an_interp * rate_ratio
                elif frac_remapped > 1e-9: # Initial was zero, remapped is non-zero
                    # This indicates remapping *caused* mergers. The rate is hard to estimate without normalization.
                    # Let's assign a value proportional to frac_remapped and analytical scale?
                    # Or maybe skip these points as potentially unreliable estimate?
                    sim_rate_est = rate_an_interp * frac_remapped / 1e-6 # Arbitrary small initial fraction guess
                    print(f"Warning: F_initial near zero for {pair_key}, f={f_val}. Sim rate estimate might be inaccurate.")
                else: # Both zero
                    sim_rate_est = 0.0

                pair_sim_f.append(f_val)
                pair_sim_rate.append(sim_rate_est)

            except Exception as e:
                print(f"Error loading or processing {sim_file}: {e}")
        else:
            print(f"Simulation result file not found: {sim_file}")

    simulation_f_values[pair_key] = np.array(pair_sim_f)
    simulation_rates[pair_key] = np.array(pair_sim_rate)

print("... Simulation Rates Estimated.")


# --- Create Plot ---
print("Generating Plot...")
plt.figure(figsize=(10, 8))

# Plot Total Analytical Rate
plt.loglog(f_pbh_values_plot, analytical_total_rates, color='white', linestyle='-', linewidth=2, label='Total Analytical Rate (Integrated)')

# Plot Analytical and Simulation Pair Rate Densities
for m1_sim, m2_sim in mass_pairs_sim:
    pair_key = f"{m1_sim:.1e}-{m2_sim:.1e}"
    color = colors.get(pair_key, 'gray') # Default color if key mismatch
    label = labels.get(pair_key, pair_key)

    # Analytical Rate Density for the pair
    if len(analytical_pair_rates[pair_key]) == len(f_pbh_values_plot):
        plt.loglog(f_pbh_values_plot, analytical_pair_rates[pair_key], color=color, linestyle='--', linewidth=1.5, label=f'{label} (Analytical Density)')

    # Simulation-Based Rate Density for the pair
    if len(simulation_f_values[pair_key]) > 0:
        plt.loglog(simulation_f_values[pair_key], simulation_rates[pair_key], color=color, linestyle='-', marker='o', markersize=5, linewidth=1.5, label=f'{label} (Simulation w/ Remapping)')
    else:
        print(f"No simulation data to plot for pair {pair_key}")


# Add LIGO/Virgo Bound Region (Example)
lvk_min = 10
lvk_max = 200
plt.fill_between(f_pbh_values_plot, lvk_min, lvk_max, color='gray', alpha=0.3, label='LVK O3 Rate Range (Approx)')


# --- Customize Plot ---
plt.xlim(1.e-4, 1.)
plt.ylim(1e-2, 1e6) # Adjust based on results
plt.xlabel(r'$f_{\rm PBH}$')
plt.ylabel(r'Merger Rate [Gpc$^{-3}$ yr$^{-1}$]')
plt.title('PBH Merger Rate: Analytical vs. Simulation (Unequal Mass)')
plt.legend(loc='lower right', fontsize='small')
plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray')
plt.tight_layout()

# Save plot
plot_filename = "pbh_merger_rate_comparison_unequal.png"
plt.savefig(plot_filename)
print(f"Comparison plot saved to {plot_filename}")
# plt.show() # Uncomment to display plot interactively

print("\nPlotting Script Finished.")
