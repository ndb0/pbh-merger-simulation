import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import argparse
import os

def create_animation(data_file, trail_frames=5):
    """
    Creates an MP4 animation from N-body position data with fixed camera
    and improved visuals.
    """
    print(f"✅ Loading position data from '{data_file}'...")
    try:
        positions_history = np.load(data_file)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found. Run 'nbody_visualizer.py' first.")
        return

    n_frames, n_particles, _ = positions_history.shape
    print(f"-> Data loaded: {n_frames} frames, {n_particles} particles.")

    # --- Robust Axis Limits ---
    initial_positions = positions_history[0]
    box_min = initial_positions.min()
    box_max = initial_positions.max()
    buffer = (box_max - box_min) * 0.1
    lim_min, lim_max = box_min - buffer, box_max + buffer

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_zlim(lim_min, lim_max)

    ax.set_xlabel('X [Mpc]', color='white')
    ax.set_ylabel('Y [Mpc]', color='white')
    ax.set_zlabel('Z [Mpc]', color='white')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # CORRECTED: Use the modern matplotlib API for 3D panes
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.grid(False)

    # Use a smaller subset of particles if there are too many, for performance
    num_to_plot = min(n_particles, 500)
    plot_indices = np.random.choice(n_particles, num_to_plot, replace=False)

    # Main particle scatter plot
    scatter = ax.scatter([], [], [], s=5, c='cyan', alpha=1.0)
    # Trail scatter plots
    trails = [ax.scatter([], [], [], s=1, c='cyan') for _ in range(trail_frames)]

    def update(frame):
        # Update main particles
        current_pos = positions_history[frame, plot_indices, :]
        scatter._offsets3d = (current_pos[:, 0], current_pos[:, 1], current_pos[:, 2])

        # Update trails
        for i in range(trail_frames):
            trail_frame_index = max(0, frame - (i + 1))
            trail_pos = positions_history[trail_frame_index, plot_indices, :]
            trails[i]._offsets3d = (trail_pos[:, 0], trail_pos[:, 1], trail_pos[:, 2])
            # Fade out the trail
            trails[i].set_alpha(0.5 * (1 - (i+1)/trail_frames))

        ax.set_title(f'PBH N-Body Evolution (Frame {frame+1}/{n_frames})', color='white')
        return [scatter] + trails

    print("🌠 Creating animation... (this may take a while)")
    ani = FuncAnimation(fig, update, frames=tqdm(range(n_frames)),
                        blit=False, interval=50)

    output_filename = 'pbh_evolution.mp4'
    ani.save(output_filename, writer='ffmpeg', fps=20, dpi=150, savefig_kwargs={'facecolor': 'black'})
    print(f"✅ Animation saved to '{output_filename}'")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an MP4 animation from N-body position data.")
    parser.add_argument('data_file', type=str, help='Path to the positions_history.npy file.')
    args = parser.parse_args()
    create_animation(args.data_file)

