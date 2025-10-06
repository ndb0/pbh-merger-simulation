import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cosmo

def plot_rate_vs_time(result_file):
    """
    Loads merger history binned in redshift, converts it to a merger rate
    as a function of cosmic time, and compares it with the theoretical
    t^(-34/37) scaling.
    """
    print(f"✅ Loading data from '{result_file}'...")
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Result file not found. Run a simulation first.")
        return

    # --- 1. Load Simulation Data (binned in redshift) ---
    z_hist = np.array(data['merger_z_hist'])
    z_bin_edges = np.array(data['merger_z_bin_edges'])
    f_pbh = data['f_pbh']

    if z_hist.sum() == 0:
        print("No merger events found. Nothing to plot.")
        return

    # --- 2. Convert Redshift Bins to Time Bins ---
    t_bin_edges_s = cosmo.t_of_a(cosmo.a_of_z(z_bin_edges))
    t_bin_edges_gyr = t_bin_edges_s / (cosmo.YEAR_S * 1e9)

    # CORRECTED: Duration must be positive. Use np.abs().
    dt_yr = np.abs(t_bin_edges_s[1:] - t_bin_edges_s[:-1]) / cosmo.YEAR_S

    t_bin_centers_gyr = (t_bin_edges_gyr[:-1] + t_bin_edges_gyr[1:]) / 2

    # --- 3. Calculate Merger Rate R(t) from Simulation ---
    # Add a small epsilon to the denominator to prevent division by zero.
    simulated_rate_t = z_hist / (dt_yr + 1e-99)

    # --- 4. Filter for Bins with Mergers for Log Plotting ---
    positive_mask = simulated_rate_t > 0
    t_plot = t_bin_centers_gyr[positive_mask]
    rate_plot = simulated_rate_t[positive_mask]

    if len(rate_plot) == 0:
        print("No positive merger rates found after processing. Cannot plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the simulated merger rate R(t)
    ax.step(t_plot, rate_plot, where='mid',
            label=f'My Simulation (f_pbh = {f_pbh})', lw=2.5, color='royalblue')

    # --- 5. Plot Theoretical t^(-34/37) Curve ---
    t_theory_gyr = np.logspace(np.log10(t_plot[0]), np.log10(t_plot[-1]), 200)
    rate_theory = t_theory_gyr**(-34/37)

    # Normalize the theoretical curve to match the simulation's first data point
    norm_factor = rate_plot[0] / rate_theory[0]
    ax.plot(t_theory_gyr, rate_theory * norm_factor,
            label='Theoretical Prediction ~ t^(-34/37)', # Use ASCII chars to avoid font warnings
            linestyle='--', color='crimson', lw=2)

    ax.set_xlabel('Cosmic Time [Gyr]', fontsize=12)
    ax.set_ylabel('Normalized Merger Rate [dN/dt]', fontsize=12)
    ax.set_title('Comparison of Merger Rate Evolution with Theory', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-", color='0.85')

    # Save the plot
    output_dir = os.path.dirname(result_file)
    plot_filename = os.path.join(output_dir, f'rate_vs_time_comparison_fpbh_{f_pbh:.1e}.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"\n✅ Plot of Rate vs. Time saved to '{plot_filename}'")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot merger rate vs. time and compare with theory.")
    parser.add_argument('result_file', type=str, help='Path to the mc_merger_history.json file.')
    args = parser.parse_args()
    plot_rate_vs_time(args.result_file)

