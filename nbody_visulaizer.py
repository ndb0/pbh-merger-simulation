import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from config import load_config
from pbh_population import generate_pbh_population
from dynamics import compute_accelerations

def run_nbody_simulation(cfg, num_steps=500, dt_myr=0.5):
    """
    Runs a simple N-body simulation to generate position data for visualization.
    Note: This is a Newtonian simulation in a static box and does not include
    cosmological expansion or GW decay. It's for visualization purposes.
    """
    start_time = time.time()

    # --- 1. Setup ---
    device = torch.device("cpu") # N-body is heavy, run on CPU unless you have a powerful GPU
    if cfg.compute.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    print(f"✅ N-body visualization run starting on {device}.")

    # --- 2. Generate Population with Velocities ---
    # We must virialize the cluster to see interesting gravitational dynamics.
    N = int(cfg.pbh_population.number_density * cfg.pbh_population.volume)
    masses, positions, velocities = generate_pbh_population(
        N=N, mu=cfg.pbh_population.f_pbh_mu, sigma=cfg.pbh_population.f_pbh_sigma,
        volume=cfg.pbh_population.volume, seed=cfg.pbh_population.seed,
        device=device, virialize=True
    )
    print(f"-> Generated {N} virialized PBHs.")

    # Convert dt from Mega-years to seconds
    dt_s = dt_myr * 1e6 * 3.154e7
    # We need to convert velocities from km/s to Mpc/s for the integrator
    velocities_mpc_s = velocities / 1e6 / 3.262 # km/s -> Mpc/s approx

    # --- 3. N-Body Evolution Loop ---
    positions_history = [positions.cpu().numpy()]
    a = compute_accelerations(masses, positions)

    for _ in tqdm(range(num_steps), desc="N-Body Evolution"):
        # Velocity Verlet Integrator
        positions = positions + velocities_mpc_s * dt_s + 0.5 * a * dt_s**2
        a_new = compute_accelerations(masses, positions)
        velocities_mpc_s = velocities_mpc_s + 0.5 * (a + a_new) * dt_s
        a = a_new
        positions_history.append(positions.cpu().numpy())

    # --- 4. Save Results ---
    output_dir = cfg.output.save_path
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'positions_history.npy')
    np.save(output_file, np.array(positions_history))

    print(f"\n✅ N-body data saved to '{output_file}'")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an N-body simulation for visualization.")
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    cfg = load_config(args.config_file)
    run_nbody_simulation(cfg)
