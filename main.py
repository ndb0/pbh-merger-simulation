import torch
import numpy as np
import os
import time
import argparse  # Import the argparse library
from config import load_config
from pbh_population import generate_pbh_population
from dynamics import update_angular_momentum_j
import cosmo

def run_simulation(config_path):
    """
    Main simulation driver.
    """
    cfg = load_config(config_path)
    start_time = time.time()

    # --- 1. Setup Compute Device ---
    if cfg.compute.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✅ Simulation configured to run on GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("✅ Simulation configured to run on CPU.")

    # --- 2. Create Time Grid ---
    t_grid, a_grid, z_grid = cosmo.get_time_grid(
        variable=cfg.time.variable,
        start=cfg.time.start,
        end=cfg.time.end,
        steps=cfg.time.steps
    )
    print(f"Time grid created with {len(t_grid)} steps, from z~{z_grid[0]:.1f} to z={z_grid[-1]:.1f}")

    # --- 3. Generate Initial PBH Population ---
    print("Generating initial PBH population...")
    N = int(cfg.pbh_population.number_density * cfg.pbh_population.volume)
    masses, positions, velocities = generate_pbh_population(
        N=N,
        mu=cfg.pbh_population.f_pbh_mu,
        sigma=cfg.pbh_population.f_pbh_sigma,
        volume=cfg.pbh_population.volume,
        seed=cfg.pbh_population.seed,
        device=device
    )
    print(f"-> Generated {N} PBHs on device '{device}'.")

    # --- 4. Mock Evolution Loop ---
    print("\n--- Running a test of the GW orbital decay function ---")
    if N >= 2:
        mock_m1 = masses[0].unsqueeze(0)
        mock_m2 = masses[1].unsqueeze(0)
        mock_a = torch.tensor([1e-7], device=device) # Initial separation in Mpc
        mock_e = torch.tensor([0.7], device=device)
        mock_j = torch.sqrt(1.0 - mock_e**2)
        dt_seconds = 3.154e7 # One year

        print(f"Initial state: a={mock_a.item():.2e} Mpc, e={mock_e.item():.3f}, j={mock_j.item():.3f}")
        j_new, a_new, e_new = update_angular_momentum_j(mock_j, mock_a, mock_e, mock_m1, mock_m2, dt_seconds)
        print(f"State after 1 year: a={a_new.item():.2e} Mpc, e={e_new.item():.3f}, j={j_new.item():.3f}")
    else:
        print("-> Skipping GW test, need at least 2 PBHs.")

    # --- 5. Save Results ---
    output_path = cfg.output.save_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, "initial_population.pt")
    torch.save({
        'masses': masses.cpu(),
        'positions': positions.cpu(),
        'velocities': velocities.cpu(),
    }, output_file)
    print(f"\n✅ Initial population state saved to {output_file}")

    end_time = time.time()
    print(f"Total script execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # --- Corrected argument parsing ---
    # Create a parser to read command-line arguments
    parser = argparse.ArgumentParser(description="Run a PBH simulation.")
    # Add an argument for the config file path
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the YAML configuration file (e.g., input_local_cpu.yaml).'
    )
    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Pass the provided config file path to the simulation function
    run_simulation(args.config_file)

