import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.backends.backend_pdf

# Set plotting style for clarity
plt.style.use('ggplot')

def load_data(filename='three_body_lookup_unequal_final.npz'):
    """Loads the pre-computed data from the NPZ file."""
    if not os.path.exists(filename):
        print(f"Error: Lookup file '{filename}' not found.")
        return None

    data = np.load(filename, allow_pickle=True)

    if 'r_evolution' not in data or 'time_grid' not in data:
        print("Error: Missing 'r_evolution' or 'time_grid' keys in NPZ file.")
        return None

    results = {key: data[key] for key in data.files}
    results['time_grid'] = results['time_grid'][0]

    # Correctly handle boolean conversion from float arrays
    for key in ['is_bound', 'is_merged']:
        if key in results:
            if results[key].ndim > 1:
                 results[key] = np.array([arr[0] > 0.5 for arr in results[key]], dtype=bool)
            else:
                 results[key] = results[key] > 0.5

    return results

def plot_orbital_evolution(data, sim_index, pdf=None):
    """
    Plots the separation |r1 - r2| as a function of time, focusing on initial dynamics.
    """

    if data is None or sim_index >= len(data['r_evolution']):
        return

    # Extract evolution arrays
    r_evolution = data['r_evolution'][sim_index]
    time_grid = data['time_grid']

    # Get parameters for the title
    params = {k: data[k][sim_index] for k in data.keys() if len(data[k]) == len(data['r_evolution'])}

    # Convert time to years and separation to AU
    time_years = time_grid / cosmo.YEAR_S
    r_au = r_evolution / (cosmo.parsec / 206265)

    # --- CRITICAL PLOTTING FIX: Use linear scale over a short period ---
    # Find a period to plot (e.g., the first 1000 years, or the first 5% of the total steps)
    time_limit_index = np.where(time_years > 1000)[0]
    plot_end_index = time_limit_index[0] if len(time_limit_index) > 0 else NUM_TIMESTEPS - 1

    time_plot = time_years[:plot_end_index]
    r_plot = r_au[:plot_end_index]

    # Normalize by the initial separation for relative comparison
    r_initial_au = r_au[0]

    # --- Create the plot ---
    plt.figure(figsize=(12, 6))

    # Plot normalized separation
    plt.plot(time_plot, r_plot / r_initial_au, label='Separation $|r| / r_0$', linewidth=2)

    # Add labels and status
    title = (
        f"Simulation {sim_index}: $r_0/r_{{max}}={params['r_initial']:.6f}$, $d/r_0={params['d_perturber']:.2f}$, "
        f"$m_3/M_{{bin}}={params['q_perturber']:.2e}$\n"
        f"Status: Bound={params['is_bound']}, Merged={params['is_merged']} | Final $j$={params['j_final']:.3e}"
    )
    plt.title(title, fontsize=10)
    plt.xlabel('Time (Years)')
    plt.ylabel('Orbital Separation Normalized to Initial Separation ($r_0$)')
    # Use Linear X-axis to see oscillations clearly
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(which='both', linestyle='--', alpha=0.5)

    if pdf:
        pdf.savefig(plt.gcf())
        plt.close()
    else:
        plt.show()

def plot_survival_map(data, pdf=None):
    """
    Plots the final state (j_final) vs. initial conditions (d/r and q)
    to visualize the disruption boundary.
    """
    if data is None: return

    q_perturber = data['q_perturber']
    d_ratio = data['d_perturber']
    j_final = data['j_final']
    is_bound = data['is_bound']

    # We filter data into categories
    unbound_mask = ~is_bound
    bound_mask = is_bound

    plt.figure(figsize=(10, 8))

    # 1. Plot Disrupted (Unbound) systems
    plt.scatter(
        d_ratio[unbound_mask],
        q_perturber[unbound_mask],
        c='red',
        marker='x',
        label='Disrupted (Unbound)',
        s=100
    )

    # 2. Plot Bound (Survived) systems, color-coded by final angular momentum (j)
    sc = plt.scatter(
        d_ratio[bound_mask],
        q_perturber[bound_mask],
        c=j_final[bound_mask],
        cmap='viridis',
        marker='o',
        label='Survived (Bound)',
        s=50,
        norm=plt.Normalize(vmin=0, vmax=1)
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Binary Survival vs. Perturber Mass and Distance')
    plt.xlabel('Perturber Distance Ratio $d / r_{\\text{initial}}$ (Log Scale)')
    plt.ylabel('Perturber Mass Ratio $m_3 / M_{\\text{binary}}$ (Log Scale)')

    # Add color bar for angular momentum
    cbar = plt.colorbar(sc, label='Final Dimensionless Angular Momentum ($j$)')
    cbar.set_label('Final j (Eccentricity $e = \sqrt{1-j^2}$)')

    plt.grid(which='both', linestyle='--', alpha=0.3)
    plt.legend()

    if pdf:
        pdf.savefig(plt.gcf())
        plt.close()
    else:
        plt.show()

def save_plots_to_pdf(data, indices_to_plot, filename="three_body_plots_final.pdf"):
    """Saves multiple plots to a single PDF document."""

    if not data: return

    # Initialize PDF backend
    with matplotlib.backends.backend_pdf.PdfPages(filename) as pdf:
        print(f"\n--- Saving Orbital Evolution to {filename} ---")

        # 1. Save individual evolution plots
        for index in indices_to_plot:
            if index < len(data['r_evolution']):
                plot_orbital_evolution(data, index, pdf)
                print(f"Saved evolution plot for index {index}.")
            else:
                print(f"Warning: Index {index} skipped (out of bounds).")

        # 2. Save the final statistical map
        print("Saving Disruption Map.")
        plot_survival_map(data, pdf)

        print(f"✅ All plots successfully saved to {filename}")


if __name__ == '__main__':
    # Define placeholder constant used in the plotting function
    NUM_TIMESTEPS = 2500 # Must match the value used in precompute.py

    try:
        import cosmo
    except ImportError:
        print("Error: Could not import 'cosmo.py'. Please ensure it is in the directory.")
        exit()

    data = load_data()

    if data is not None:
        print(f"Loaded {len(data['r_evolution'])} simulation results.")

        # --- Find the best run for visualization ---
        r_a_final = data['r_a_final']

        # CRASH FIX: Filter the array to ensure it is not empty before calling argmin
        valid_r_a = r_a_final[r_a_final > 0]

        if valid_r_a.size == 0:
            # If no bound binaries were found, fallback to plotting the first few indices.
            best_run_index = 0
            print("WARNING: No bound binaries found. Plotting initial/default index 0.")
        else:
            # Find the index in the original r_a_final array corresponding to the minimum value
            min_r_a = np.min(valid_r_a)
            original_indices = np.where(r_a_final == min_r_a)[0]
            best_run_index = original_indices[0] # Use the first occurrence
            print(f"Best run for visualization (tightest orbit) found at Index: {best_run_index}")

        # --- Define which runs to save ---
        indices_to_save = [
            best_run_index, # The tightest orbit found (or fallback)
            0, # The loosest initial conditions
            len(data['r_evolution']) - 1 # The tightest initial conditions (last index)
        ]

        # Remove duplicates and ensure indices are valid integers
        indices_to_save = sorted(list(set(i % len(data['r_evolution']) for i in indices_to_save)))

        # Call the new PDF saving function
        save_plots_to_pdf(data, indices_to_save, filename="three_body_plots_final.pdf")
