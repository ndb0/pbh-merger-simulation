import torch
import numpy as np
import os
import time
import argparse
import json
from tqdm import tqdm
from config import load_config
from pbh_population import generate_pbh_population
from dynamics import update_angular_momentum_j
import cosmo

def run_full_simulation(cfg):
    """
    Runs a time-stepped simulation of PBH binary evolution.
    This tracks the GW-induced decay of all potential pairs over cosmic time.
    """
    start_time = time.time()

    # --- 1. Load Config and Setup Device ---
    try:
        f_pbh = cfg.pbh_population.f_pbh
        mu_solar = cfg.pbh_population.f_pbh_mu
        output_dir = cfg.output.save_path
    except AttributeError as e:
        print(f"❌ Configuration Error: Missing key in YAML file: {e}")
        return

    device = torch.device("cpu")
    if cfg.compute.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    print(f"✅ Simulation starting on {device}. Config: {output_dir}")

    # --- 2. Generate Initial Population ---
    N = int(cfg.pbh_population.number_density * cfg.pbh_population.volume)
    masses, positions, _ = generate_pbh_population(
        N=N, mu=mu_solar, sigma=cfg.pbh_population.f_pbh_sigma,
        volume=cfg.pbh_population.volume, seed=cfg.pbh_population.seed,
        device=device, virialize=False
    )
    print(f"-> Generated {N} PBHs.")

    # --- 3. Setup Time Grid for Evolution ---
    t_grid_s, _, z_grid = cosmo.get_time_grid(
        variable=cfg.time.variable, start=cfg.time.start,
        end=cfg.time.end, steps=cfg.time.steps
    )
    print(f"-> Time grid created with {len(t_grid_s)} steps, from z~{z_grid[0]:.1f} to z={z_grid[-1]:.1f}")

    # --- 4. Identify Initial Binaries and Their Properties ---
    print("Identifying all initial pairs and their orbital properties...")
    p_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    r_ij = torch.norm(p_diff, dim=2)
    indices = torch.triu(torch.ones_like(r_ij, dtype=torch.bool), diagonal=1).nonzero()
    idx1, idx2 = indices[:, 0], indices[:, 1]

    m1, m2 = masses[idx1], masses[idx2]
    M_solar = m1 + m2
    eta = (m1 * m2) / M_solar**2
    a = r_ij[idx1, idx2] / 2.0  # Semi-major axis for zero-velocity capture

    j = 1.7e-2 * (M_solar)**(5/37) * (4 * eta)**(3/37) * f_pbh**(10/37)
    e = torch.sqrt(torch.clamp(1.0 - j**2, min=0.0))

    active_binaries = {'m1': m1, 'm2': m2, 'a': a, 'e': e, 'j': j,
                       'merged': torch.zeros_like(m1, dtype=torch.bool)}
    print(f"-> Tracking the evolution of {len(idx1)} potential binaries.")

    # --- 5. Main Evolution Loop ---
    merger_events = []
    total_mergers = 0
    for i in tqdm(range(len(t_grid_s) - 1), desc="Evolving Binaries"):
        dt = t_grid_s[i+1] - t_grid_s[i]
        if dt <= 0: continue

        not_merged_mask = ~active_binaries['merged']
        if not not_merged_mask.any():
            print("All binaries have merged. Ending simulation early.")
            break

        original_indices = not_merged_mask.nonzero().flatten()

        m1_active, m2_active, a_active, e_active, j_active = (
            active_binaries['m1'][not_merged_mask], active_binaries['m2'][not_merged_mask],
            active_binaries['a'][not_merged_mask], active_binaries['e'][not_merged_mask],
            active_binaries['j'][not_merged_mask]
        )

        j_new, a_new, e_new = update_angular_momentum_j(
            j_active, a_active, e_active, m1_active, m2_active, dt
        )

        original_dtype = active_binaries['a'].dtype
        active_binaries['a'][not_merged_mask] = a_new.to(original_dtype)
        active_binaries['e'][not_merged_mask] = e_new.to(original_dtype)
        active_binaries['j'][not_merged_mask] = j_new.to(original_dtype)

        merger_this_step = a_new < 1e-20
        if merger_this_step.any():
            current_z = z_grid[i]
            merged_indices_in_active = merger_this_step.nonzero().flatten()
            total_mergers += len(merged_indices_in_active)

            merged_original_indices = original_indices[merged_indices_in_active]

            for original_idx in merged_original_indices:
                merger_events.append({
                    'm1_solar': active_binaries['m1'][original_idx].item(),
                    'm2_solar': active_binaries['m2'][original_idx].item(),
                    'redshift': current_z,
                    'time_gyr': t_grid_s[i+1] / (cosmo.YEAR_S * 1e9)
                })

            active_binaries['merged'][merged_original_indices] = True

            # Use carriage return to print on the same line below the progress bar
            tqdm.write(f"   z={current_z:.2f}: {len(merged_indices_in_active)} new mergers! (Total: {total_mergers})")

    # --- 6. Save Results ---
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f'simulation_history_fpbh_{f_pbh:.1e}.json')
    with open(output_filename, 'w') as f:
        class ConfigEncoder(json.JSONEncoder):
            def default(self, o):
                return o.__dict__
        json.dump({'merger_events': merger_events, 'config': cfg}, f, indent=4, cls=ConfigEncoder)

    print(f"\n✅ Simulation complete. Merger history saved to '{output_filename}'")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a time-stepped PBH evolution simulation.")
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    cfg = load_config(args.config_file)
    run_full_simulation(cfg)

