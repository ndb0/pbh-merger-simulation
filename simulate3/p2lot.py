import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cosmo # Import cosmo for constants
from config import load_config # Import load_config

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
        # Load new mass keys
        m1_solar = data['m1_solar']
        m2_solar = data['m2_solar']
        M_total_solar = data['M_total_solar']
        f_pbh = data['f_pbh']
        eta = data['eta']
    except Exception as e:
        print(f"Error loading data from .npz file: {e}")
        return

    # Get age of universe for plotting
    t_universe_yrs = cosmo.T_0_S / cosmo.YEAR_S

    # --- Plotting ---
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define bins for the histogram (logarithmic)
    # Filter out potential inf/nan values from t_initial_yrs
    valid_t_initial = t_initial_yrs[np.isfinite(t_initial_yrs) & (t_initial_yrs > 0)]
    if valid_t_initial.size == 0:
        print("Error: No valid initial merger times to plot.")
        return

    bins = np.logspace(np.log10(valid_t_initial.min()), np.log10(valid_t_initial.max()), 50)

    # Plot histogram for initial merger times (blue)
    ax.hist(
        t_initial_yrs,
        bins=bins,
        alpha=0.7,
        label=f'Initial Binaries (N={len(t_initial_yrs)})',
        color='blue',
        log=True,
        histtype='step',
        linewidth=2
    )

    # Plot histogram for final (remapped) merger times (red)
    ax.hist(
        t_final_yrs,
        bins=bins,
        alpha=0.7,
        label=f'Remapped Binaries (N={len(t_final_yrs)})',
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

    # Use wide but sensible limits
    min_time = max(1e9, valid_t_initial.min())
    max_time = min(1e11, valid_t_initial.max())
    ax.set_xlim(min_time, max_time)
    ax.set_ylim(bottom=1) # Start y-axis at 1 for log plot

    output_filename = os.path.splitext(data_file)[0] + '.png'
    plt.savefig(output_filename)
    print(f"✅ Plot saved to {output_filename}")
    # plt.show() # Comment out to prevent blocking in some environments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PBH merger time distribution.")
    # Argument is now optional
    parser.add_argument('config_file', type=str, nargs='?', default='input_local_cpu.yaml',
                        help='(Optional) Path to the YAML config file to find the .npz file. Defaults to input_local_cpu.yaml')
    args = parser.parse_args()

    data_file = None

    # Try to find the default output file based on the config
    print(f"Loading config from {args.config_file} to find data file...")
    try:
        cfg = load_config(args.config_file)
        f_pbh = cfg.pbh_population.f_pbh
        # Use new mass keys
        m1_solar = float(cfg.pbh_population.m1_solar)
        m2_solar = float(cfg.pbh_population.m2_solar)
        M_total_solar = m1_solar + m2_solar
        output_dir = cfg.output.save_path

        # CRASH FIX: Use the correct scientific notation format for the filename
        default_file = os.path.join(output_dir, f'statistical_merger_times_M_total_{M_total_solar:.2e}_f{f_pbh:.0e}.npz')

        if os.path.exists(default_file):
            print(f"Found data file: {default_file}")
            data_file = default_file
        else:
            print(f"Error: Default data file not found at {default_file}")
            # Try falling back to old f_pbh_mu key just in case
            if hasattr(cfg.pbh_population, 'f_pbh_mu'):
                M_PBH_old = float(cfg.pbh_population.f_pbh_mu)
                default_file_old = os.path.join(output_dir, f'statistical_merger_times_M_total_{M_PBH_old:.2e}_f{f_pbh:.0e}.npz')
                if os.path.exists(default_file_old):
                     print(f"Found old-format data file: {default_file_old}")
                     data_file = default_file_old
                else:
                     print(f"Error: Also could not find old-format file at {default_file_old}")
                     exit()
            else:
                exit()

    except Exception as e:
        print(f"Error: Could not load config or find data file. {e}")
        exit()

    plot_merger_time_distribution(data_file)

