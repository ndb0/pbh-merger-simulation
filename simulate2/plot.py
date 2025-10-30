import numpy as np
import matplotlib.pyplot as plt
import os

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

    # Convert bools/floats stored as arrays of arrays
    for i in range(len(results['is_bound'])):
        # Since these are stored as float64 arrays by numpy.savez_compressed,
        # we check the float value for boolean status.
        results['is_bound'][i] = results['is_bound'][i] > 0.5
        results['is_merged'][i] = results['is_merged'][i] > 0.5

    return results

def plot_orbital_evolution(data, sim_index):
    """
    Plots the separation |r1 - r2| as a function of time for a single simulation.
    """

    if not data or sim_index >= len(data['r_evolution']):
        print(f"Error: Simulation index {sim_index} out of bounds.")
        return

    # Extract evolution arrays
    r_evolution = data['r_evolution'][sim_index]
    time_grid = data['time_grid']

    # Get parameters for the title
    params = {k: data[k][sim_index] for k in data.keys() if len(data[k]) == len(data['r_evolution'])}

    # Convert time to years for readability
    time_years = time_grid / cosmo.YEAR_S

    # Convert separation to AU for scale (approx)
    r_au = r_evolution / (cosmo.parsec / 206265)

    # Calculate the mean separation (r_mean) by taking the average of the non-zero time series
    r_mean = np.mean(r_au[r_au > 0])

    plt.figure(figsize=(12, 6))

    # Plot normalized separation
    plt.plot(time_years, r_au / r_mean, label=f'Separation |r| / $\\langle r \\rangle$')

    # Add status lines
    title = (
        f"Simulation {sim_index}: r/r_max={params['r_initial']:.2f}, d/r={params['d_perturber']:.2f}, "
        f"m3/M={params['q_perturber']:.2e}\n"
        f"Status: Bound={params['is_bound']}, Merged={params['is_merged']} | Final $j$={params['j_final']:.2e}"
    )
    plt.title(title, fontsize=10)
    plt.xlabel('Time (Years)')
    plt.ylabel('Orbital Separation Normalized to Mean $\\langle r \\rangle$')
    plt.yscale('log') # Use log scale to see wide range of dynamics
    plt.grid(which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

def plot_survival_map(data):
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
    plt.xlabel('Perturber Distance Ratio $d / r_{\\text{initial}}$')
    plt.ylabel('Perturber Mass Ratio $m_3 / M_{\\text{binary}}$')

    # Add color bar for angular momentum
    cbar = plt.colorbar(sc, label='Final Dimensionless Angular Momentum ($j$)')
    cbar.set_label('Final j (Eccentricity $e = \sqrt{1-j^2}$)')

    plt.grid(which='both', linestyle='--', alpha=0.3)
    plt.legend()
    plt.show()

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

        # --- Orbital Evolution Plots (Verification) ---
        print("\n--- Orbital Evolution Plots (Verification) ---")

        # Plot 3 specific examples:
        # NOTE: Index 0 is often undisturbed. We'll use other indices for diversity.

        # Run 1: Undisrupted (Index 0)
        print("Plotting Run 1 (Index 0): Should show simple oscillation.")
        plot_orbital_evolution(data, 0)

        # Run 2: Perturbed (Index 50)
        print("Plotting Run 2 (Index 50): Should show a perturbed or expanding orbit.")
        plot_orbital_evolution(data, 50)

        # Run 3: Disrupted/Ionized (Index 100)
        print("Plotting Run 3 (Index 100): Should show a rapid change or unbound final state.")
        plot_orbital_evolution(data, 100)

        # --- Statistical Survival Map ---
        print("\n--- Disruption Map (Statistical Analysis) ---")
        plot_survival_map(data)
