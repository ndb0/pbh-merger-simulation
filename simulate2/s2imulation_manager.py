import torch
import numpy as np
import time
from tqdm import tqdm
from config import load_config
import cosmo
import nbody_kernels
import pbh_population
import merger_physics
import os

def find_binaries(positions, velocities, masses, G):
    """
    Analyzes the final state of an N-body simulation and finds all bound binaries.
    """
    N = masses.shape[0] # [int]
    device = positions.device
    binaries = []

    # Create all unique pairs (i, j) where i < j
    for i in range(N):
        for j in range(i + 1, N):
            m1 = masses[i, 0] # [kg]
            m2 = masses[j, 0] # [kg]
            M_total = m1 + m2 # [kg]

            r_vec = positions[j] - positions[i] # [m]
            r_norm = torch.norm(r_vec) # [m]

            v_vec = velocities[j] - velocities[i] # [m s^-1]
            v_norm_sq = torch.dot(v_vec, v_vec) # [m^2 s^-2]

            # 1. Check if bound (Total Energy < 0)
            mu = (m1 * m2) / M_total # [kg]
            E_kin = 0.5 * mu * v_norm_sq # [J]
            E_pot = -G * m1 * m2 / r_norm # [J]
            E_total = E_kin + E_pot # [J]

            if E_total < 0:
                # Calculate orbital parameters
                L_vec = mu * torch.cross(r_vec, v_vec) # [kg m^2 s^-1]
                L_norm = torch.norm(L_vec) # [kg m^2 s^-1]

                # Calculate r_a. Only valid if bound (E_total < 0)
                r_a = -G * m1 * m2 / (2.0 * E_total) # [m]

                # If r_a calculation is numerically unstable, skip
                if not torch.isfinite(r_a) or r_a <= 0:
                    continue

                # Calculate j
                j_denom = mu * torch.sqrt(G * M_total * r_a) # [kg m^2 s^-1]
                j = L_norm / j_denom # [dimensionless]
                j = torch.clamp(j, min=0.0, max=1.0) # Clamp numerical errors

                # Calculate coalescence time (tau)
                tau_s = merger_physics.calculate_coalescence_time(r_a, j, m1, m2, G, cosmo.c) # [s]

                binaries.append({
                    'm1_kg': m1.item(), 'm2_kg': m2.item(),
                    'r_a_m': r_a.item(), 'j': j.item(),
                    'tau_s': tau_s.item() if isinstance(tau_s, torch.Tensor) else tau_s # [s]
                })

    return binaries


def run_stage_1_early_universe(cfg):
    """
    RUNS STAGE 1: Simulates many small N-body boxes in the early universe.
    """
    print("--- STAGE 1: EARLY UNIVERSE SIMULATION ---")

    # --- 1. Setup ---
    device = torch.device("cuda" if cfg.compute.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SI constants
    G = cosmo.G # [m^3 kg^-1 s^-2]
    a_eq = cosmo.a_eq # [dimensionless]
    t_eq = cosmo.t_eq # [s]

    # Simulation parameters
    N_sims = cfg.stage1.num_simulations # [int]
    N_particles_per_sim = cfg.stage1.num_particles # [int]

    # Time grid
    z_start = 1e9 # [dimensionless]
    t_start = cosmo.t_of_a(cosmo.a_of_z(z_start)) # [s] (~24 s)
    t_end = t_eq # [s] (~65k yr)
    timesteps = cfg.stage1.timesteps # [int]

    # --- CRITICAL FIX: Use LOGARITHMIC time grid ---
    if t_start >= t_end:
         raise ValueError(f"t_start ({t_start}) must be less than t_end ({t_end}). Check cosmo.py calculations.")
    time_grid = np.geomspace(t_start, t_end, timesteps) # [s]
    # ---

    all_binaries = []

    # --- VERBOSE TIME OUTPUT ---
    dt_min = (time_grid[1] - time_grid[0]) / time_grid[0]
    dt_max = (time_grid[-1] - time_grid[-2]) / time_grid[-2]
    print(f"Time Grid Info ({timesteps} steps):")
    print(f"  Start Time: {t_start/cosmo.YEAR_S:.2e} yr (z={z_start:.1e})")
    print(f"  End Time:   {t_end/cosmo.YEAR_S:.2e} yr (z={cosmo.z_of_a(cosmo.a_of_t(t_end)):.1e})")
    print(f"  |dt/t| starts at {dt_min:.2e}, ends at {dt_max:.2e}")
    # ---------------------------

    # --- 2. Run Simulations ---
    for i in tqdm(range(N_sims), desc="Stage 1 Sims"):
        # 1. Get Initial Conditions (Prints verbose output inside pbh_population.py)
        masses, positions, velocities = pbh_population.generate_pbh_population_si(
            N=N_particles_per_sim,
            cfg=cfg,
            seed=cfg.pbh_population.seed + i,
            device=device,
            t_start_s=t_start
        )

        # 2. Set up kernel arguments
        kernel_args = {'G': G, 'a_eq': a_eq, 't_eq': t_eq}

        # 3. Evolve the system
        for k in range(timesteps - 1):
            t = time_grid[k] # [s]
            dt = time_grid[k+1] - t # [s]

            if dt <= 0: continue

            positions, velocities = nbody_kernels.leapfrog_kick_drift_kick(
                positions, velocities, masses,
                nbody_kernels.get_acceleration_early_universe,
                dt, t, **kernel_args
            )

        # 4. Find and save binaries at the END of the simulation (t_eq)
        final_binaries = find_binaries(positions, velocities, masses, G)
        all_binaries.extend(final_binaries)

        # --- VERBOSE BINARY OUTPUT ---
        print(f"[Sim {i+1}/{N_sims}] Found {len(final_binaries)} bound binaries. Total found so far: {len(all_binaries)}")
        # -----------------------------

    # --- 3. Save Results ---
    print(f"\n--- STAGE 1 COMPLETE ---")
    print(f"Total binaries found across {N_sims} simulations: {len(all_binaries)}")
    return all_binaries


def run_stage_2_late_universe(cfg, initial_binary_catalog):
    """
    RUNS STAGE 2: Simulates many collapsed halos in the late universe.
    """
    print("\n--- STAGE 2: LATE UNIVERSE HALO SIMULATION ---")

    # ... (rest of Stage 2 simulation logic) ...
    device = torch.device("cuda" if cfg.compute.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    G = cosmo.G # [m^3 kg^-1 s^-2]
    c = cosmo.c # [m s^-1]

    # Simulation parameters
    N_halos = cfg.stage2.num_halos # [int]
    N_particles_per_halo = cfg.stage2.num_particles # [int]
    halo_radius_pc = cfg.stage2.halo_radius_pc # [pc]
    halo_radius_m = halo_radius_pc * cosmo.parsec # [m]

    # Time grid (from z=z_eq to z=0)
    t_start = cosmo.t_eq # [s]
    t_end = cosmo.T_0_S  # Age of universe # [s]
    timesteps = cfg.stage2.timesteps # [int]
    time_grid = np.linspace(t_start, t_end, timesteps) # [s]

    merger_log = [] # Logs generation, time, masses
    disruption_log = [] # Logs S_L

    # --- 2. Run Simulations ---
    for i in tqdm(range(N_halos), desc="Stage 2 Sims"):
        # 1. Get Initial Conditions for this halo
        masses, positions, velocities, spins, binary_info = \
            pbh_population.generate_halo_population_si(
                N=N_particles_per_halo,
                mu_kg=cfg.pbh_population.f_pbh_mu * cosmo.M_sun_SI, # [kg]
                sigma=cfg.pbh_population.f_pbh_sigma, # [dimensionless]
                radius_m=halo_radius_m, # [m]
                seed=cfg.pbh_population.seed + i,
                device=device,
                initial_binary_catalog=initial_binary_catalog,
                G=G
            )

        # 2. Set up kernel arguments
        kernel_args = {'G': G}

        # 3. Evolve the system
        for k in range(timesteps - 1):
            t = time_grid[k] # [s]
            dt = time_grid[k+1] - t # [s]

            if dt <= 0: continue

            # 3a. HIERARCHICAL PHYSICS: Detect and perform mergers
            new_mergers, new_disruptions, state_updates = \
                merger_physics.detect_and_handle_events(
                    positions, velocities, masses, spins, binary_info,
                    dt, t, G, c
                )

            merger_log.extend(new_mergers)

            # Apply state updates (remove merged BHs, add new ones)
            if state_updates and (state_updates.get('remove') or state_updates.get('add')):
                 positions, velocities, masses, spins, binary_info = \
                    pbh_population.update_simulation_state(
                        positions, velocities, masses, spins, binary_info,
                        state_updates
                    )

            # Check if simulation still has particles
            if masses.shape[0] < 2:
                 break

            # 3b. N-BODY STEP: Evolve the (potentially updated) system
            positions, velocities = nbody_kernels.leapfrog_kick_drift_kick(
                positions, velocities, masses,
                nbody_kernels.get_acceleration_late_universe,
                dt, t, **kernel_args
            )

    # --- 3. Save Results ---
    print(f"\n--- STAGE 2 COMPLETE ---")
    print(f"Total mergers logged across {N_halos} halos: {len(merger_log)}")
    S_L = 1.0 # Placeholder
    print(f"S_L (Late Survival) = {S_L:.4f} (Placeholder)")

    return merger_log, S_L

if __name__ == "__main__":
    # 0. Load Configuration
    config_path = 'input_local_cpu.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at: {config_path}")
        exit()

    cfg = load_config(config_path)

    # --- Configuration Modification Recommendation ---
    # The user wants 10^6 particles. We set this as the target N_USER_INTENDED.
    N_USER_INTENDED = 1000000

    # Initialize default values if missing (to prevent the AttributeError)

    # CRASH FIX: Ensure we have access to ConfigNamespace if needed
    try:
        from config import ConfigNamespace
    except ImportError:
        # Define a minimal ConfigNamespace if config.py is not available/working
        class ConfigNamespace:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

    # CRASH FIX: Initialize missing sections immediately
    if not hasattr(cfg, 'stage1'):
        cfg.stage1 = ConfigNamespace({'num_simulations': 10, 'num_particles': 50, 'timesteps': 10000})
        print("NOTICE: 'stage1' config missing in YAML. Using internal defaults.")

    if not hasattr(cfg, 'stage2'):
        cfg.stage2 = ConfigNamespace({'num_halos': 10, 'num_particles': 1000000, 'halo_radius_pc': 10, 'timesteps': 10000})
        print("NOTICE: 'stage2' config missing in YAML. Using internal defaults.")

    if not hasattr(cfg.pbh_population, 'seed'):
        cfg.pbh_population.seed = 42

    if not hasattr(cfg, 'compute'):
        cfg.compute = ConfigNamespace({'use_gpu': False})


    # --- Particle Count Adjustment ---
    # Since the user requested 10^6 but we must start small for debugging:
    N_SIM_MAX_SAFE = 5000

    if cfg.stage1.num_particles != N_USER_INTENDED:
        print(f"\n[ATTENTION: SCALING]")
        print(f"  User requested N={N_USER_INTENDED}. For stability and debugging the current physics,")
        # Ensure we don't crash by using a huge N if the user changed the YAML
        if cfg.stage1.num_particles > N_SIM_MAX_SAFE:
            cfg.stage1.num_particles = N_SIM_MAX_SAFE
            print(f"  WARNING: Reducing N to safe max N={N_SIM_MAX_SAFE} for stability.")
        print(f"  Running Stage 1 with N={cfg.stage1.num_particles} to find physical binaries.")

    # -------------------------------------


    # 1. Run Stage 1 to get the initial binary population
    start_time = time.time()
    initial_binary_catalog = run_stage_1_early_universe(cfg)
    print(f"Stage 1 total execution time: {time.time() - start_time:.2f} s")

    if not initial_binary_catalog:
        print("No binaries found in Stage 1. Exiting.")
    else:
        # 2. Run Stage 2 to evolve halos and find mergers
        start_time = time.time()
        merger_log, S_L_factor = run_stage_2_late_universe(cfg, initial_binary_catalog)
        print(f"Stage 2 total execution time: {time.time() - start_time:.2f} s")

        # 3. Final Analysis
        print("\n--- FINAL RESULTS ---")
        print(f"Total hierarchical mergers recorded: {len(merger_log)}")
        generations = {}
        if merger_log:
            for m in merger_log:
                gen = m.get('generation', 'unknown')
                generations[gen] = generations.get(gen, 0) + 1
            for gen, count in sorted(generations.items()):
                 print(f"  Generation {gen} mergers: {count}")
        else:
            print("  No mergers occurred in Stage 2.")
