import torch
import numpy as np
import time
from tqdm import tqdm
from config import load_config
import cosmo
import nbody_kernels
import pbh_population
import merger_physics

def find_binaries(positions, velocities, masses, G):
    """
    Analyzes the final state of an N-body simulation and finds all bound binaries.
    This is how the "Initial Binary Catalog" is built.
    
    Args:
        positions (torch.Tensor): Shape (N, 3)
        velocities (torch.Tensor): Shape (N, 3)
        masses (torch.Tensor): Shape (N, 1)
        G (float): Gravitational constant
        
    Returns:
        list: A list of dictionaries, where each dict represents a stable binary.
              e.g., [{'m1': 1.0, 'm2': 1.2, 'r_a': 1e10, 'j': 0.1, ...}]
    """
    print("Finding bound binaries...")
    N = masses.shape[0]
    device = positions.device
    binaries = []
    
    m_expanded = masses.view(N, 1)
    
    # Create all unique pairs (i, j) where i < j
    for i in range(N):
        for j in range(i + 1, N):
            m1 = masses[i, 0]
            m2 = masses[j, 0]
            M_total = m1 + m2
            
            r_vec = positions[j] - positions[i]
            r_norm = torch.norm(r_vec)
            
            v_vec = velocities[j] - velocities[i]
            v_norm_sq = torch.dot(v_vec, v_vec)
            
            # 1. Check if bound (Kinetic Energy < Potential Energy)
            # E_kin = 0.5 * mu * v^2, where mu = m1*m2 / (m1+m2)
            # E_pot = -G * m1 * m2 / r
            # We are bound if E_kin + E_pot < 0
            # 0.5 * (m1*m2 / M_total) * v^2 - G * m1 * m2 / r < 0
            # 0.5 * v^2 < G * M_total / r
            
            if 0.5 * v_norm_sq < (G * M_total / r_norm):
                # This pair is bound! Now, calculate its orbital parameters.
                
                # 2. Calculate angular momentum vector: L = mu * (r x v)
                mu = (m1 * m2) / M_total
                L_vec = mu * torch.cross(r_vec, v_vec)
                L_norm = torch.norm(L_vec)
                
                # 3. Calculate semi-major axis 'a' (or r_a)
                # E = -G * m1 * m2 / (2 * a)
                # E = E_kin + E_pot = 0.5 * mu * v_norm_sq - G * m1 * m2 / r_norm
                # -G * m1 * m2 / (2 * a) = 0.5 * mu * v_norm_sq - G * m1 * m2 / r_norm
                # -G * M_total / (2 * a) = 0.5 * v_norm_sq - G * M_total / r_norm
                # 1 / a = 2 * (G * M_total / r_norm - 0.5 * v_norm_sq) / (G * M_total)
                # 1 / a = (2 / r_norm) - (v_norm_sq / (G * M_total))
                r_a = 1.0 / ( (2.0 / r_norm) - (v_norm_sq / (G * M_total)) )
                
                # 4. Calculate dimensionless angular momentum 'j' (from Eq. 4)
                # j = (L / mu) / sqrt(r_a * M_total * G)  (G=1 in paper, but we need it)
                j = (L_norm / mu) / torch.sqrt(r_a * M_total * G)
                
                # 5. Calculate coalescence time (tau)
                tau_s = merger_physics.calculate_coalescence_time(r_a, j, m1, m2, G, cosmo.c)
                
                binaries.append({
                    'm1_kg': m1.item(),
                    'm2_kg': m2.item(),
                    'r_a_m': r_a.item(),
                    'j': j.item(),
                    'tau_s': tau_s.item()
                })
                
    print(f"Found {len(binaries)} bound binaries.")
    return binaries

def run_stage_1_early_universe(cfg):
    """
    RUNS STAGE 1: Simulates many small N-body boxes in the early universe.
    This numerically computes the initial binary population (and thus S_E).
    """
    print("--- STAGE 1: EARLY UNIVERSE SIMULATION ---")
    
    # --- 1. Setup ---
    device = torch.device("cuda" if cfg.compute.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load SI constants
    G = cosmo.G
    a_eq = cosmo.a_eq
    t_eq = cosmo.t_eq
    
    # Simulation parameters
    N_sims = cfg.stage1.num_simulations
    N_particles_per_sim = cfg.stage1.num_particles
    sim_volume_m3 = (cfg.stage1.box_size_mpc * cosmo.Mpc)**3
    
    # Time grid
    # We simulate from z=10^9 down to z=z_eq
    z_start = 1e9
    t_start = cosmo.t_of_a(cosmo.a_of_z(z_start))
    t_end = t_eq
    timesteps = cfg.stage1.timesteps
    time_grid = np.linspace(t_start, t_end, timesteps)
    
    all_binaries = []
    
    # --- 2. Run Simulations ---
    for i in tqdm(range(N_sims), desc="Stage 1 Sims"):
        # 1. Get Initial Conditions
        # We must use the *fixed* SI unit versions from pbh_population.py
        masses, positions, velocities = pbh_population.generate_pbh_population_si(
            N=N_particles_per_sim,
            mu_kg=cfg.pbh_population.f_pbh_mu * cosmo.M_sun_SI,
            sigma=cfg.pbh_population.f_pbh_sigma,
            volume_m3=sim_volume_m3,
            seed=cfg.pbh_population.seed + i,
            device=device,
            t_start_s=t_start
        )
        
        # 2. Set up kernel arguments
        kernel_args = {'G': G, 'a_eq': a_eq, 't_eq': t_eq}
        
        # 3. Evolve the system
        for k in range(timesteps - 1):
            t = time_grid[k]
            dt = time_grid[k+1] - t
            
            positions, velocities = nbody_kernels.leapfrog_kick_drift_kick(
                positions, velocities, masses,
                nbody_kernels.get_acceleration_early_universe,
                dt, t, **kernel_args
            )
            
        # 4. Find and save binaries
        final_binaries = find_binaries(positions, velocities, masses, G)
        all_binaries.extend(final_binaries)
        
    # --- 3. Save Results ---
    print(f"\n--- STAGE 1 COMPLETE ---")
    print(f"Total binaries found across {N_sims} simulations: {len(all_binaries)}")
    # In a real project, you would save `all_binaries` to a file (e.g., HDF5, Parquet)
    # This file is your "Initial Binary Catalog"
    return all_binaries


def run_stage_2_late_universe(cfg, initial_binary_catalog):
    """
    RUNS STAGE 2: Simulates many collapsed halos in the late universe.
    This numerically computes S_L, late 2/3-body channels, and hierarchical mergers.
    """
    print("\n--- STAGE 2: LATE UNIVERSE HALO SIMULATION ---")
    
    # --- 1. Setup ---
    device = torch.device("cuda" if cfg.compute.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    G = cosmo.G
    c = cosmo.c
    
    # Simulation parameters
    N_halos = cfg.stage2.num_halos
    N_particles_per_halo = cfg.stage2.num_particles
    halo_radius_pc = cfg.stage2.halo_radius_pc
    halo_radius_m = halo_radius_pc * cosmo.parsec
    
    # Time grid (from z=z_eq to z=0)
    t_start = cosmo.t_eq
    t_end = cosmo.T_0_S  # Age of universe
    timesteps = cfg.stage2.timesteps
    time_grid = np.linspace(t_start, t_end, timesteps)
    
    # This will be our main output
    merger_log = [] # Logs generation, time, masses
    disruption_log = [] # Logs S_L

    # --- 2. Run Simulations ---
    for i in tqdm(range(N_halos), desc="Stage 2 Sims"):
        # 1. Get Initial Conditions for this halo
        #    This is a complex step. For now, we just sample N particles
        #    and "inject" binaries from our catalog.
        
        # TODO: A real implementation needs a Halo Mass Function to pick halo sizes
        # and a model to populate them.
        
        # For now, a simple example:
        masses, positions, velocities, spins, binary_info = \
            pbh_population.generate_halo_population_si(
                N=N_particles_per_halo,
                mu_kg=cfg.pbh_population.f_pbh_mu * cosmo.M_sun_SI,
                sigma=cfg.pbh_population.f_pbh_sigma,
                radius_m=halo_radius_m,
                seed=cfg.pbh_population.seed + i,
                device=device,
                initial_binary_catalog=initial_binary_catalog,
                G=G
            )
        
        # 2. Set up kernel arguments
        kernel_args = {'G': G}
        
        # 3. Evolve the system
        for k in range(timesteps - 1):
            t = time_grid[k]
            dt = time_grid[k+1] - t
            
            # 3a. HIERARCHICAL PHYSICS: Detect and perform mergers
            new_mergers, new_disruptions, state_updates = \
                merger_physics.detect_and_handle_events(
                    positions, velocities, masses, spins, binary_info,
                    dt, t, G, c
                )
            
            # Log events
            merger_log.extend(new_mergers)
            disruption_log.extend(new_disruptions)
            
            # Apply state updates (remove merged BHs, add new ones)
            if state_updates:
                positions, velocities, masses, spins, binary_info = \
                    pbh_population.update_simulation_state(
                        positions, velocities, masses, spins, binary_info,
                        state_updates
                    )
            
            # 3b. N-BODY STEP: Evolve the (potentially updated) system
            positions, velocities = nbody_kernels.leapfrog_kick_drift_kick(
                positions, velocities, masses,
                nbody_kernels.get_acceleration_late_universe,
                dt, t, **kernel_args
            )
            
    # --- 3. Save Results ---
    print(f"\n--- STAGE 2 COMPLETE ---")
    print(f"Total mergers logged: {len(merger_log)}")
    print(f"Total disruptions logged: {len(disruption_log)}")
    
    # Calculate S_L (Late disruption suppression)
    initial_binaries_tracked = sum(1 for d in disruption_log)
    surviving_binaries = sum(1 for d in disruption_log if d['survived'])
    S_L = surviving_binaries / initial_binaries_tracked if initial_binaries_tracked > 0 else 1.0
    print(f"S_L (Late Survival) = {S_L:.4f}")
    
    # This log is your final data product
    return merger_log, S_L

if __name__ == "__main__":
    # This is a conceptual workflow
    
    # 0. Load Configuration
    # We need to add new keys to our YAML for this new sim
    # e.g., stage1.num_simulations, stage2.num_halos
    # For now, we'll use the existing file
    cfg = load_config('input_local_cpu.yaml')

    # --- Add conceptual new config keys ---
    # These should be in your YAML file!
    cfg.stage1 = type('',(object,),{
        'num_simulations': 10,  # Small number for a test
        'num_particles': 50,    # Small number for a test
        'box_size_mpc': 0.01,   # 10 kpc box
        'timesteps': 100
    })()
    cfg.stage2 = type('',(object,),{
        'num_halos': 10,       # Small number for a test
        'num_particles': 100,
        'halo_radius_pc': 10,
        'timesteps': 1000
    })()
    # -------------------------------------

    
    # 1. Run Stage 1 to get the initial binary population
    # This numerically calculates S_E by only finding binaries that *actually* form
    start_time = time.time()
    initial_binary_catalog = run_stage_1_early_universe(cfg)
    print(f"Stage 1 took {time.time() - start_time:.2f} s")

    if not initial_binary_catalog:
        print("No binaries found in Stage 1. Exiting.")
    else:
        # 2. Run Stage 2 to evolve halos and find mergers
        # This numerically calculates S_L, late channels, and hierarchical mergers
        start_time = time.time()
        merger_log, S_L_factor = run_stage_2_late_universe(cfg, initial_binary_catalog)
        print(f"Stage 2 took {time.time() - start_time:.2f} s")

        # 3. Final Analysis (replaces montc_sim.py's histogram)
        # You would now analyze 'merger_log' to build your histogram
        print("\n--- FINAL RESULTS ---")
        print(f"Total hierarchical mergers recorded: {len(merger_log)}")
        for i in range(1, 4):
            gen_i_mergers = sum(1 for m in merger_log if m['generation'] == i)
            print(f"  Generation {i} mergers: {gen_i_mergers}")

