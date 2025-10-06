import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_merger_history(result_file):
    """
    Loads merger history from a Monte Carlo simulation and plots the
    merger rate density vs. redshift.
    """
    print(f"✅ Loading data from '{result_file}'...")
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Result file not found. Run 'monte_carlo_mergers.py' first.")
        return

    hist = np.array(data['merger_z_hist'])
    bin_edges = np.array(data['merger_z_bin_edges'])
    f_pbh = data['f_pbh']

    if hist.sum() == 0:
        print("No merger events found in the results file. Nothing to plot.")
        return

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize histogram to get a rate density
    # This is a relative rate, not an absolute one in Gpc^-3 yr^-1 yet
    rate_density = hist / np.sum(hist)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.step(bin_centers, rate_density, where='mid',
            label=f'f_pbh = {f_pbh}', lw=2)

    ax.set_xlabel('Redshift (z)', fontsize=12)
    ax.set_ylabel('Normalized Merger Rate Density [dN/dz]', fontsize=12)
    ax.set_title('Distribution of PBH Merger Events over Cosmic Time', fontsize=14)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=1e-4) # Adjust if needed

    # Save the plot
    output_dir = os.path.dirname(result_file)
    plot_filename = os.path.join(output_dir, f'merger_history_plot_fpbh_{f_pbh:.1e}.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"\n✅ Plot saved to '{plot_filename}'")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot merger history from a Monte Carlo simulation.")
    parser.add_argument('result_file', type=str, help='Path to the mc_merger_history.json file.')
    args = parser.parse_args()
    plot_merger_history(args.result_file)

''''
### **3. Your New Workflow**

This new approach is much faster and more robust.

**Step 1: Run the Monte Carlo Simulation**
This will be very fast as it's not a step-by-step evolution.
```bash
# Activate your environment
conda activate pbh_sim

# Run the Monte Carlo simulation
python monte_carlo_mergers.py input_local_cpu.yaml
```
This will create a new results file, e.g., `results_local_cpu/mc_merger_history_fpbh_1.0e-02.json`.

**Step 2: Plot the Merger History**
Use the new plotting script to visualize the merger distribution.
```bash
python plot_merger_history.py results_local_cpu/mc_merger_history_fpbh_1.0e-02.json
'''
