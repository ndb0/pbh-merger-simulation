import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cosmo

def theoretical_rate_density_vs_z(z):
    """
    Calculates the theoretical merger rate DENSITY dN/dz as a function of redshift.
    This is the physically correct quantity to compare with a histogram in z.
    """
    # 1. Get time 't' corresponding to redshift 'z'
    a = cosmo.a_of_z(z)
    t_s = cosmo.t_of_a(a)

    # 2. Calculate the rate R(t), which is proportional to t^(-34/37)
    # We will normalize this later, so we only need the time-dependent part.
    rate_vs_t = (t_s / cosmo.T_0_S)**(-34/37)

    # 3. Calculate the conversion factor |dt/dz| = 1 / (H(z) * (1+z))
    # This comes from the chain rule: dt/dz = dt/da * da/dz
    dt_dz = 1.0 / (cosmo.H_z(z) * (1 + z))

    # 4. The merger rate density is R(t) * |dt/dz|
    rate_density_vs_z = rate_vs_t * dt_dz

    return rate_density_vs_z

def plot_merger_history(result_file):
    """
    Loads merger history and plots both the simulated data and the
    corrected theoretical prediction on the same graph.
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
        print("No merger events found. Nothing to plot.")
        return

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize the simulation histogram to get a probability density
    bin_width = bin_edges[1] - bin_edges[0]
    simulated_rate_density = hist / (hist.sum() * bin_width)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot simulated data
    ax.step(bin_centers, simulated_rate_density, where='mid',
            label=f'My Simulation (f_pbh = {f_pbh})', lw=2.5, color='royalblue')

    # Calculate and plot theoretical prediction
    z_theory = np.linspace(0.1, bin_centers.max(), 200)
    rate_density_theory = theoretical_rate_density_vs_z(z_theory)

    # Normalize the theoretical curve to match the area of the simulation histogram
    # This allows comparing the *shape* of the distributions.
    norm_factor = np.sum(simulated_rate_density * bin_width) / np.trapz(rate_density_theory, z_theory)
    ax.plot(z_theory, rate_density_theory * norm_factor,
            label='Corrected Theory (dN/dz)',
            linestyle='--', color='crimson', lw=2)

    ax.set_xlabel('Redshift (z)', fontsize=12)
    ax.set_ylabel('Normalized Merger Rate Density [dN/dz]', fontsize=12)
    ax.set_title('Comparison of Simulated Merger History with Theory', fontsize=14)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlim(left=0, right=20)
    ax.set_ylim(bottom=1e-3) # Adjust if needed

    # Save the plot
    output_dir = os.path.dirname(result_file)
    plot_filename = os.path.join(output_dir, f'comparison_merger_history_fpbh_{f_pbh:.1e}.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"\n✅ Comparison plot saved to '{plot_filename}'")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot merger history and compare with theory.")
    parser.add_argument('result_file', type=str, help='Path to the mc_merger_history.json file.')
    args = parser.parse_args()
    plot_merger_history(args.result_file)

