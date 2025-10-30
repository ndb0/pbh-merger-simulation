import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cosmo # Import cosmo for constants
from config import load_config # Import config loader

def plot_merger_time_distribution(data_file):
    """
    Loads the .npz file from the statistical simulation and plots
    a histogram of the initial vs. final merger times.
    """
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    try:
        data = np.load(data_file)
        t_initial_yrs = data['t_initial_yrs']
        t_final_yrs = data['t_final_yrs']
        # Load the new unequal mass keys
        m1_solar = data['m1_solar']
        m2_solar = data['m2_solar']
        f_pbh = data['f_pbh']
    except Exception as e:
        print(f"Error loading data from .npz file: {e}")
        return

    # Get age of universe for plotting
    t_universe_yrs = cosmo.T_0_S / cosmo.YEAR_S

    # --- Plotting ---
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define bins for the histogram (logarithmic)
    # Filter out potential inf/nan values
    t_initial_valid = t_initial_yrs[np.isfinite(t_initial_yrs) & (t_initial_yrs > 0)]
    t_final_valid = t_final_yrs[np.isfinite(t_final_yrs) & (t_final_yrs > 0)]

    if len(t_initial_valid) == 0 or len(t_final_valid) == 0:
        print("Error: No valid merger time data to plot.")
        return

    min_bin = min(t_initial_valid.min(), t_final_valid.min())
    max_bin = max(t_initial_valid.max(), t_final_valid.max())

    # Handle potential edge case where min/max are the same or invalid
    if min_bin <= 0 or max_bin <= 0 or min_bin >= max_bin:
        min_bin = 1e9
        max_bin = 1e11

    bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 50)

    # Plot histogram for initial merger times (blue)
    ax.hist(
        t_initial_valid,
        bins=bins,
        alpha=0.7,
        label=f'Initial Binaries (N={len(t_initial_valid)})',
        color='blue',
        log=True,
        histtype='step',
        linewidth=2
    )

    # Plot histogram for final (remapped) merger times (red)
    ax.hist(
        t_final_valid,
        bins=bins,
        alpha=0.7,
        label=f'Remapped Binaries (N={len(t_final_valid)})',
        color='red',
        log=True,
        histtype='step',
        linewidth=2,
        linestyle='dashed'
    )

    # Add a vertical line for the age of the universe
    ax.axvline(
        t_universe_yrs,
        color='black',
        linestyle='--',
        linewidth=2,
        label=f'Age of Universe ({t_universe_yrs:.1e} yrs)'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Merger Time (Years)')
    ax.set_ylabel('Number of Binaries')
    # Update title to show both masses
    ax.set_title(f'PBH Merger Time Distribution (m1={m1_solar} M_sun, m2={m2_solar} M_sun, f={f_pbh:.0e})')
    ax.legend()
    ax.set_xlim(1e9, max_bin) # Focus on the relevant time range
    ax.set_ylim(bottom=1) # Start y-axis at 1 for log plot

    output_filename = os.path.splitext(data_file)[0] + '.png'
    plt.savefig(output_filename)
    print(f"✅ Plot saved to {output_filename}")
    # plt.show() # Comment out to prevent blocking in some environments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PBH merger time distribution.")
    # Require config file
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file.')
    # Make data file optional
    parser.add_argument('data_file', type=str, nargs='?', default=None,
                        help='(Optional) Path to the .npz file. If not provided, tries to find it from config.')
    args = parser.parse_args()

    data_file = args.data_file
    cfg = load_config(args.config_file)

    # If no file provided, try to find the default output
    if data_file is None:
        print("No data file provided. Trying to find default output from config...")
        try:
            f_pbh = cfg.pbh_population.f_pbh
            m1_solar = float(cfg.pbh_population.m1_solar)
            m2_solar = float(cfg.pbh_population.m2_solar)
            M_total_solar = m1_solar + m2_solar
            output_dir = cfg.output.save_path

            # CRASH FIX: Use the same scientific notation formatting as the saving script
            default_file = os.path.join(output_dir, f'statistical_merger_times_M_total_{M_total_solar:.2e}_f{f_pbh:.0e}.npz')

            if os.path.exists(default_file):
                print(f"Found data file: {default_file}")
                data_file = default_file
            else:
                print(f"Error: Default data file not found at {default_file}")
                # Try the *other* format just in case
                old_format_file = os.path.join(output_dir, f'statistical_merger_times_M_total_{M_total_solar}_f{f_pbh:.0e}.npz')
                if os.path.exists(old_format_file):
                     print(f"Found file with alternate format: {old_format_file}")
                     data_file = old_format_file
                else:
                    print(f"Also could not find: {old_format_file}")
                    exit()
        except Exception as e:
            print(f"Error: Could not load config to find default data file. {e}")
            exit()


    plot_merger_time_distribution(data_file)

