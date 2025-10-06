import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from config import load_config
from pbh_population import generate_pbh_population
from dynamics import compute_accelerations
import time

def run_nbody_simulation(cfg):
    """
    Runs a simplified N-body simulation and saves the position history.
    """
    start_time = time.time()

    # --- 1. Load Config and Setup ---
    try:
        n_density = cfg.pbh_population.number_density
        volume = cfg.pbh_population.volume
        output_dir = cfg.output.save_path
    except AttributeError as e:
        print(f"❌ Configuration Error: Missing key in YAML file: {e}")
        return

    print(f"✅ N-body visualization starting on CPU. Config: {output_dir}")

    # --- 2. Generate Initial Population ---
    N = int(n_density * volume)
    masses, positions, velocities = generate_pbh_population(
        N=N,
        mu=cfg.pbh_population.f_pbh_mu,
        sigma=cfg.pbh_population.f_pbh_sigma,
        volume=volume,
        seed=cfg.pbh_population.seed,
        device='cpu',
        virialize=True  # Virialized for interesting dynamics
    )
    print(f"-> Generated {N} PBHs.")

    # --- 3. Simulation Parameters ---
    n_steps = 200  # Number of steps to save for the video
    dt_s = 1e14      # Timestep in seconds (chosen for visualization)

    # Conversion factor from seconds to astronomical time units [Mpc/(km/s)]
    # This is needed to make the integrator dimensionally consistent.
    s_to_astro_time = 1 / 3.086e19
    dt = dt_s * s_to_astro_time

    positions_history = np.zeros((n_steps, N, 3))

    # --- 4. N-Body Evolution Loop (Corrected Leapfrog Integrator) ---
    print("-> Evolving system (this may take a while)...")

    # Initial half-step for velocity
    a = compute_accelerations(masses, positions)
    velocities += a * (dt / 2.0)

    for i in tqdm(range(n_steps)):
        # Full-step for position
        positions += velocities * dt

        # Full-step for velocity
        a = compute_accelerations(masses, positions)
        velocities += a * dt

        positions_history[i] = positions.cpu().numpy()

    # --- 5. Save Results ---
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, 'positions_history.npy')

    np.save(output_filename, positions_history)

    print(f"\n✅ N-body simulation complete. Position history saved to '{output_filename}'")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an N-body simulation for visualization.")
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    cfg = load_config(args.config_file)
    run_nbody_simulation(cfg)

