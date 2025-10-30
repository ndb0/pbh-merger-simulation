import numpy as np
import matplotlib.pyplot as plt
import os
import cosmo # Import cosmo for constants

def plot_oscillation_data(data_file):
    """
    Loads the .npz file from the N-body oscillation test
    and plots the separation vs. time to check for orbits.
    """
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    try:
        data = np.load(data_file, allow_pickle=True)
        r_evolution_m = data['r_evolution']
        time_grid_s = data['time_grid_s']
        is_bound = data['is_bound']
        j_final = data['j_final']
        r_initial_ratio = data['r_initial_ratio']
    except Exception as e:
        print(f"Error loading data from .npz file: {e}")
        return

    print(f"Plotting results for {data_file}...")
    print(f"Final Status: Bound={is_bound}, j_final={j_final:.2e}")

    # --- Plotting ---
    plt.style.use('ggplot')

    # Convert time to seconds (relative to start) and separation to AU
    time_s = time_grid_s - time_grid_s[0]
    r_au = r_evolution_m / (cosmo.parsec / 206265)

    # --- Figure 1: The First Few Orbits (Linear Time) ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Plot only the first 50 orbital periods to see oscillations
    # (T_orb_si was ~0.03s in the simulation script)
    T_orb_approx = 0.03 # seconds
    plot_duration_s = 50 * T_orb_approx # Plot ~1.5 seconds

    # Find the index corresponding to this duration
    plot_end_index = np.where(time_s > plot_duration_s)[0]
    if len(plot_end_index) > 0:
        idx = plot_end_index[0]
    else:
        idx = len(time_s) - 1 # Fallback to end of array

    ax1.plot(time_s[:idx], r_au[:idx], label=f'Separation |r|')

    ax1.set_title(f'Initial Binary Capture (First {plot_duration_s:.1f} seconds)')
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel('Separation (AU)')
    ax1.grid(which='both', linestyle='--', alpha=0.5)
    ax1.legend()

    plt.tight_layout()
    plt.savefig('oscillation_plot_zoomed_in.png')
    print(f"✅ Zoomed-in oscillation plot saved to oscillation_plot_zoomed_in.png")

    # --- Figure 2: The Full Evolution (Log Time) ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    time_yrs = time_s / cosmo.YEAR_S

    ax2.plot(time_yrs, r_au, label='Separation |r|')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Time (Years)')
    ax2.set_ylabel('Separation (AU)')
    ax2.set_title(f'Full Orbital Evolution (r0/rmax = {r_initial_ratio:.1e})')
    ax2.grid(which='both', linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('oscillation_plot_full.png')
    print(f"✅ Full evolution plot saved to oscillation_plot_full.png")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3-body oscillation data.")
    parser.add_argument('data_file', type=str, nargs='?', default='oscillation_test_run.npz',
                        help='(Optional) Path to the .npz file. Defaults to oscillation_test_run.npz')
    args = parser.parse_args()

    plot_oscillation_data(args.data_file)
