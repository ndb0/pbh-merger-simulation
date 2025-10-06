import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import argparse
from tqdm import tqdm

def create_animation(position_file):
    """
    Loads position history and creates a 3D animated scatter plot.
    """
    print(f"✅ Loading position data from '{position_file}'...")
    try:
        positions = np.load(position_file)
    except FileNotFoundError:
        print(f"❌ Error: Position data file not found. Run 'nbody_visualizer.py' first.")
        return

    num_frames, num_particles, _ = positions.shape
    print(f"-> Data loaded: {num_frames} frames, {num_particles} particles.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set consistent axis limits based on the initial positions
    box_lim = positions[0].max()
    ax.set_xlim(0, box_lim)
    ax.set_ylim(0, box_lim)
    ax.set_zlim(0, box_lim)
    ax.set_xlabel('X [Mpc]')
    ax.set_ylabel('Y [Mpc]')
    ax.set_zlabel('Z [Mpc]')
    ax.set_title('PBH N-Body Evolution')

    # Initialize the scatter plot
    scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], s=2, alpha=0.7)

    # Animation update function
    def update(frame):
        # Update the positions of the points
        scatter._offsets3d = (positions[frame, :, 0], positions[frame, :, 1], positions[frame, :, 2])
        ax.set_title(f'PBH N-Body Evolution (Frame {frame+1}/{num_frames})')
        return scatter,

    # Create the animation
    print("🎬 Creating animation... (this may take a while)")
    anim = FuncAnimation(fig, update, frames=tqdm(range(num_frames), desc="Rendering Frames"),
                         blit=True, interval=30)

    # Save the animation
    output_path = "pbh_evolution.mp4"
    anim.save(output_path, writer='ffmpeg', fps=30, dpi=150)
    print(f"\n✅ Animation saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an MP4 animation from N-body position data.")
    parser.add_argument('position_file', type=str, help='Path to the positions_history.npy file.')
    args = parser.parse_args()
    create_animation(args.position_file)
