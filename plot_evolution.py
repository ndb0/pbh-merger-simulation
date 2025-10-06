import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_merger_history(results_file):
    """
    Loads simulation history data and plots the merger rate as a function of redshift.
    """
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Results file not found at '{results_file}'")
        return

    merger_events = data.get('merger_events', [])
    if not merger_events:
        print("No merger events found in the results file. Nothing to plot.")
        return

    print(f"✅ Loaded {len(merger_events)} merger events from '{results_file}'.")

    # Extract redshift for each merger
    redshifts = [event['redshift'] for event in merger_events]

    # --- Create a histogram of mergers vs redshift ---
    fig, ax = plt.subplots(figsize=(12, 7))

    bins = np.logspace(np.log10(max(0.1, min(redshifts))), np.log10(max(redshifts) + 1), 50)
    ax.hist(redshifts, bins=bins, color='royalblue', alpha=0.7, edgecolor='black', label='Simulated Mergers')

    ax.set_xlabel("Redshift (z)", fontsize=14)
    ax.set_ylabel("Number of Merger Events", fontsize=14)
    ax.set_title("Distribution of PBH Mergers over Cosmic Time", fontsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Use logarithmic scales for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Invert x-axis so time flows from left (high z) to right (low z)
    ax.invert_xaxis()
    ax.legend()

    # --- Save the plot ---
    output_dir = os.path.dirname(results_file)
    plot_path = os.path.join(output_dir, "merger_history_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Plot saved to '{plot_path}'")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot merger history from a simulation run.")
    parser.add_argument('results_file', type=str, help='Path to the simulation history JSON file.')
    args = parser.parse_args()
    plot_merger_history(args.results_file)
