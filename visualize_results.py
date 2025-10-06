import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_results(file_path):
    """
    Loads the simulation output file and generates plots for the
    initial mass distribution, velocity distribution, and spatial distribution.
    """
    # --- 1. Load the Data ---
    if not os.path.exists(file_path):
        print(f"❌ Error: Data file not found at '{file_path}'")
        return

    print(f"✅ Loading data from '{file_path}'...")
    data = torch.load(file_path)
    masses = data['masses'].numpy()
    positions = data['positions'].numpy()
    velocities = data['velocities'].numpy()

    print(f"-> Loaded {len(masses)} particles.")

    # --- 2. Create Figure and Subplots ---
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    fig.suptitle('PBH Simulation Initial Conditions', fontsize=16)

    # --- 3. Plot Mass Distribution ---
    ax1.hist(masses, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Mass Distribution')
    ax1.set_xlabel('Mass ($M_\\odot$)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 4. Plot Velocity Distribution ---
    # Convert velocities from Mpc/s to km/s for easier interpretation
    Mpc_to_km = 3.086e19
    velocities_kms = velocities * Mpc_to_km
    velocity_magnitudes = np.linalg.norm(velocities_kms, axis=1)

    ax2.hist(velocity_magnitudes, bins=50, color='salmon', edgecolor='black', alpha=0.7)
    ax2.set_title('Velocity Magnitude Distribution')
    ax2.set_xlabel('Velocity (km/s)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- 5. Plot 3D Spatial Distribution ---
    # For performance, only plot a subset of points if there are too many
    num_points_to_plot = min(2000, len(positions))
    indices = np.random.choice(len(positions), num_points_to_plot, replace=False)
    subset_pos = positions[indices]

    scatter = ax3.scatter(subset_pos[:, 0], subset_pos[:, 1], subset_pos[:, 2],
                          c=masses[indices], cmap='viridis', s=10, alpha=0.8)
    ax3.set_title('Spatial Distribution (Subset)')
    ax3.set_xlabel('X (Mpc)')
    ax3.set_ylabel('Y (Mpc)')
    ax3.set_zlabel('Z (Mpc)')
    cbar = fig.colorbar(scatter, ax=ax3, pad=0.1)
    cbar.set_label('Mass ($M_\\odot$)')

    # --- 6. Show and Save the Plot ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = os.path.dirname(file_path)
    plot_filename = os.path.join(output_dir, 'initial_conditions_plot.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"\n✅ Plot saved to '{plot_filename}'")

    # plt.show() # Uncomment this line if you are in a graphical environment and want to see the plot immediately

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the initial conditions from a PBH simulation."
    )
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to the .pt file containing the simulation results (e.g., results_local_cpu/initial_population.pt).'
    )
    args = parser.parse_args()
    plot_results(args.file_path)
