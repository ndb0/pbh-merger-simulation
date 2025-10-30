import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.backends.backend_pdf

# Set plotting style for clarity
plt.style.use('ggplot')

def load_data(filename='three_body_lookup_unequal.npz'):
    """Loads the pre-computed data from the NPZ file."""
    if not os.path.exists(filename):
        print(f"Error: Lookup file '{filename}' not found.")
        return None

    # Load the compressed file
    data = np.load(filename, allow_pickle=True)

    # Check that r_evolution and time_grid exist
    if 'r_evolution' not in data or 'time_grid' not in data:
        print("Error: Missing 'r_evolution' or 'time_grid' keys in NPZ file.")
        return None

    # We extract the results, ensuring boolean fields are correctly interpreted.
    results = {key: data[key] for key in data.files}

    # Time grid is stored once per run, but should be identical for all runs.
    # We take the first one.
    results['time_grid'] = results['time_grid'][0]

    # Convert arrays of float64 (where 1.0 means True) into proper boolean arrays
    # This fixes the TypeError: ufunc 'invert' not supported
    for key in ['is_bound', 'is_merged']:
        if key in results:
            # We explicitly check against the first element, which is an array,
            # to determine the correct structure for conversion.
            if results[key].ndim > 1:
                 # Case 1: Stored as an array of arrays (objects/float arrays)
                 results[key] = np.array([arr[0] > 0.5 for arr in results[key]], dtype=bool)
            else:
                 # Case 2: Stored as a simple 1D array of floats
                 results[key] = results[key] > 0.5

    return results

def plot_orbital_evolution(data, sim_index, pdf=None):
    """
    Plots the separation |r1 - r2| as a function of time for a single simulation.
    If a pdf object is provided, the figure is saved to the PDF instead of shown.
    """

    if not data or sim_index >= len(data['r_evolution']):
        # If running in a PDF loop, we don't want to print an error for every missing index
        return

    # Extract evolution arrays
    r_evolution = data['r_evolution'][sim_index]
    time_grid = data['time_grid']

    # Get parameters for the title
    params = {k: data[k][sim_index] for k in data.keys() if len(data[k]) == len(data['r_evolution'])}

    # Convert time to years for readability
    time_years = time_grid / cosmo.YEAR_S

    # Convert separation to AU for scale (approx)
    # 1 AU = cosmo.parsec / 206265
    r_au = r_evolution / (cosmo.parsec / 206265)

    # Calculate the mean separation (r_mean) by taking the average of the non-zero time series
    r_mean = np.mean(r_au[r_au > 0])

    # --- Create the plot ---
    plt.figure(figsize=(12, 6))

    # Plot normalized separation
    plt.plot(time_years, r_au / r_mean, label=f'Separation |r| / $\\langle r \\rangle$', linewidth=2)

    # Add status lines
    title = (
        f"Simulation {sim_index}: $r_0/r_{{max}}={params['r_initial']:.2f}$, $d/r_0={params['d_perturber']:.2f}$, "
        f"$m_3/M_{{bin}}={params['q_perturber']:.2e}$\n"
        f"Status: Bound={params['is_bound']}, Merged={params['is_merged']} | Final $j$={params['j_final']:.3f}"
    )
    plt.title(title, fontsize=10)
    plt.xlabel('Time (Years)')
    plt.ylabel('Orbital Separation Normalized to Mean $\\langle r \\rangle$ (Log Scale)')
    plt.xscale('log')
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

    # 1. Plot Unbound (Disrupted) systems as red X's
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
        norm=plt.Normalize(vmin=0, vmax=1) # Set fixed color range for j
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

def save_plots_to_pdf(data, indices_to_plot, filename="orbital_evolution_summary.pdf"):
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
    # CRASH FIX: Ensure the entire execution block is properly unindented
    try:
        import cosmo
    except ImportError:
        print("Error: Could not import 'cosmo.py'. Please ensure it is in the directory.")
        exit()

    data = load_data()

    if data is not None:
        print(f"Loaded {len(data['r_evolution'])} simulation results.")

        # --- Define which runs to save ---
        # We will save the three examples requested, plus the statistical map.
        indices_to_save = [0, 50, 100]

        # Call the new PDF saving function
        save_plots_to_pdf(data, indices_to_save, filename="three_body_plots_unequal.pdf")

        # Optional: Display the map interactively if not running purely for PDF generation
        # plot_survival_map(data)
