import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
import glob

def plot_merger_rates(results_dir):
    """
    Loads all .json merger rate results from a directory and plots
    the merger rate as a function of f_PBH.
    """
    # --- 1. Find and Load all Result Files ---
    search_path = os.path.join(results_dir, 'merger_rate_fpbh_*.json')
    result_files = glob.glob(search_path)

    if not result_files:
        print(f"❌ Error: No result files found in '{results_dir}'.")
        print("Please run 'calculate_merger_rate.py' for different f_PBH values first.")
        return

    print(f"✅ Found {len(result_files)} result files.")

    results = []
    for f_path in result_files:
        with open(f_path, 'r') as f:
            results.append(json.load(f))

    # Sort results by f_pbh for clean plotting
    results.sort(key=lambda x: x['f_pbh'])

    # --- 2. Extract Data for Plotting ---
    f_pbh_values = [r['f_pbh'] for r in results]
    suppressed_rates = [r['suppressed_rate_gpc_yr'] for r in results]
    raw_rates = [r['raw_rate_gpc_yr'] for r in results]

    # --- 3. Create the Plot ---
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot your simulation results
    ax.plot(f_pbh_values, suppressed_rates, 'o-', color='crimson', label='My Simulation (Suppressed)', markersize=8, lw=2)
    ax.plot(f_pbh_values, raw_rates, 's--', color='dodgerblue', label='My Simulation (Raw)', markersize=6, alpha=0.6)

    # --- 4. Add Paper's Results for Comparison ---
    # Add the LVK upper bound line from Figure 8
    ax.axhline(y=60, color='gray', linestyle='--', label='LVK Upper Bound (approx.)')

    # --- 5. Configure Plot Aesthetics ---
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$f_{PBH}$ (Fraction of Dark Matter in PBHs)', fontsize=14)
    ax.set_ylabel('Merger Rate [Gpc$^{-3}$ yr$^{-1}$]', fontsize=14)
    ax.set_title('Comparison of PBH Merger Rates', fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Set plot limits similar to the paper
    ax.set_xlim(min(f_pbh_values) * 0.5, max(f_pbh_values) * 2)
    # ax.set_ylim(1e-2, 1e4)

    # --- 6. Save and Show ---
    plot_filename = os.path.join(results_dir, 'merger_rate_vs_fpbh.png')
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)

    print(f"\n✅ Final comparison plot saved to '{plot_filename}'")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PBH merger rates vs f_PBH.")
    parser.add_argument(
        'results_dir',
        type=str,
        help='Path to the directory containing the .json result files.'
    )
    args = parser.parse_args()
    plot_merger_rates(args.results_dir)
