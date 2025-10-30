import torch
import numpy as np
import cosmo
import nbody_kernels # Assumed to contain get_acceleration_early_universe and leapfrog
import os
from tqdm import tqdm # CRASH FIX: Import tqdm for the progress bar

# --- Configuration for Pre-computation ---

# FIXED MASSES for the Inner Binary (m1 and m2) in Solar Masses (M_sun)
M1_SOLAR = 1.0
M2_SOLAR = 1e-3
M_BINARY_SOLAR = M1_SOLAR + M2_SOLAR

# PERTURBER MASS GRID (m3) - From 1e-3 M_sun to 1 M_sun
# We define the perturber mass grid directly in Solar Masses
M3_GRID_SOLAR = np.geomspace(1e-3, 1.0, 5) # 5 steps: 1e-3, ~3.1e-3, ~1e-2, ~3.1e-2, 1.0 M_sun

# SPATIAL GRIDS (Dimensionless ratios)
R_RATIO_GRID = np.linspace(0.1, 2.0, 5) # r_initial / r_max (initial separation relative to max bound radius)
D_RATIO_GRID = np.linspace(1.5, 5.0, 5) # d_perturber / r_initial (perturber distance relative to initial separation)

# SIMULATION PARAMETERS
NUM_TIMESTEPS = 2000 # High resolution for 3-body dynamics

# --- Fixed Cosmological and Physical Parameters ---
G_SI = cosmo.G
M_SUN_SI = cosmo.M_sun_SI
Z_START = 1e9
T_START = cosmo.t_of_a(cosmo.a_of_z(Z_START))
T_END = cosmo.t_eq # Simulates until matter-radiation equality
TIME_GRID = np.geomspace(T_START, T_END, NUM_TIMESTEPS)

def calculate_initial_r_max(m1_si, m2_si, t_start):
    """
    Calculates the maximum physical separation (r_max) for a pair to be gravitationally bound
    against the Hubble flow at t_start. r_max = G*M / H^2(t).
    """
    M_total = m1_si + m2_si # [kg]

    # In radiation era, H(t) = 0.5 / t
    H_start = 0.5 / t_start # [s^-1]
    H_sq = H_start**2 # [s^-2]

    # r_max (physical) is the analytical boundary for gravitational dominance
    r_max = (G_SI * M_total) / H_sq # [m]

    return r_max # [m]

def setup_three_body_si(r_initial, d_perturber, m1_si, m2_si, m3_si, t_start):
    """
    Sets up the initial state for the 3-body system at t_start.

    Args:
        r_initial (float): Separation of m1 and m2. # [m]
        d_perturber (float): Distance of m3 from the center-of-mass (CoM). # [m]
        m1_si, m2_si (float): Masses of inner binary. # [kg]
        m3_si (float): Mass of perturber. # [kg]

    Returns: (masses, positions, velocities) for 3 bodies.
    """

    masses = torch.tensor([[m1_si], [m2_si], [m3_si]], dtype=torch.float64) # [kg]

    # 1. Positions (Physical coordinates, r_initial along x-axis, perturber along y)

    # Calculate Center of Mass (CoM) relative coordinates for the inner binary
    M_inner = m1_si + m2_si
    r1_rel = -(m2_si / M_inner) * r_initial
    r2_rel = (m1_si / M_inner) * r_initial

    pos1 = torch.tensor([r1_rel, 0.0, 0.0]) # [m]
    pos2 = torch.tensor([r2_rel, 0.0, 0.0]) # [m]

    # Perturber is placed at distance 'd_perturber' from the CoM
    pos3 = torch.tensor([0.0, d_perturber, 0.0]) # [m]

    positions = torch.stack([pos1, pos2, pos3]) # [m]

    # 2. Velocities (Hubble Flow + Small Peculiar Perturbation)
    H_start = 0.5 / t_start # [s^-1]

    # Add a minimal random peculiar velocity (angular momentum)
    V_peculiar_multiplier = 1e-4 # Very small kick for initial j

    V_hubble = H_start * positions # [m/s]

    # Randomly sampled peculiar velocity vector - scaled by the largest distance in the system (d_perturber)
    V_peculiar = V_peculiar_multiplier * H_start * d_perturber * torch.randn(3, 3) # [m/s]

    velocities = V_hubble + V_peculiar # [m/s]

    # Ensure CoM velocity is zero (simplifies analysis)
    M_total_system = m1_si + m2_si + m3_si
    V_com = torch.sum(masses * velocities, dim=0) / M_total_system
    velocities = velocities - V_com # [m/s]

    return masses, positions, velocities

def run_single_three_body_sim(r_initial, d_perturber, m1_si, m2_si, m3_si, M_binary_si, log_buffer):
    """
    Runs one simulation and records the outcome (survival, disruption, merger).

    Args:
        M_binary_si (float): The total mass of the inner binary (m1 + m2). # [kg]
    """

    r_max = calculate_initial_r_max(m1_si, m2_si, T_START) # r_max for the inner binary

    # Array to log orbital separation |r1 - r2| at each step for plotting
    r_evolution = np.zeros(NUM_TIMESTEPS)

    # Set up initial conditions
    masses, positions, velocities = setup_three_body_si(
        r_initial, d_perturber, m1_si, m2_si, m3_si, T_START
    )

    # N-body simulation loop
    kernel_args = {'G': G_SI, 'a_eq': cosmo.a_eq, 't_eq': cosmo.t_eq}

    for k in range(NUM_TIMESTEPS):

        # 1. Log Separation (before the drift step)
        r_vec_log = positions[1] - positions[0]
        r_evolution[k] = torch.norm(r_vec_log).item() # [m]

        if k == NUM_TIMESTEPS - 1:
            break # Exit after logging the last position

        t = TIME_GRID[k]
        dt = TIME_GRID[k+1] - t

        positions, velocities = nbody_kernels.leapfrog_kick_drift_kick(
            positions, velocities, masses,
            nbody_kernels.get_acceleration_early_universe,
            dt, t, **kernel_args
        )

    # --- ANALYSIS OF FINAL STATE (t = t_eq) ---

    # Check the state of the inner binary (particles 0 and 1)
    m1, m2 = masses[0, 0], masses[1, 0]

    r_vec = positions[1] - positions[0]
    r_norm = torch.norm(r_vec)
    v_vec = velocities[1] - velocities[0]

    # CRASH FIX: Dot v_vec with itself to get the squared magnitude
    v_norm_sq = torch.dot(v_vec, v_vec)

    # Energy check for final bound state
    mu = (m1 * m2) / M_binary_si
    E_kin = 0.5 * mu * v_norm_sq
    E_pot = -G_SI * m1 * m2 / r_norm
    E_total = E_kin + E_pot

    # State recording
    r_a_final = 0.0
    j_final = 0.0
    is_bound = E_total < 0 # Check if inner binary is still bound
    is_merged = False

    if is_bound:
        # If bound, calculate final orbital parameters (r_a, j, tau)
        r_a_final = -G_SI * m1 * m2 / (2.0 * E_total)
        L_vec = mu * torch.cross(r_vec, v_vec)
        L_norm = torch.norm(L_vec)
        j_denom = mu * torch.sqrt(G_SI * M_binary_si * r_a_final)
        j_final = (L_norm / j_denom).item() if j_denom != 0 else 0.0

        # Calculate Coalescence Time (tau)
        tau_s = merger_physics.calculate_coalescence_time(r_a_final, j_final, m1, m2, G_SI, cosmo.c)

        # Check for merger (tau < age of universe)
        if tau_s < cosmo.T_0_S:
            is_merged = True

    # Record result to the buffer
    log_buffer.append({
        'r_initial': r_initial / r_max,
        'd_perturber': d_perturber / r_initial,
        'q_perturber': m3_si / M_binary_si,
        'is_bound': is_bound,
        'is_merged': is_merged,
        'r_a_final': r_a_final / r_max if is_bound else 0.0,
        'j_final': j_final,
        # Log the full evolution data for visualization
        'r_evolution': r_evolution,
        'time_grid': TIME_GRID
    })

    return is_bound, is_merged

if __name__ == "__main__":

    try:
        from merger_physics import calculate_coalescence_time # Used inside run_single_three_body_sim
        import merger_physics # Also needed for call
    except ImportError:
        print("ERROR: Could not import 'merger_physics.py'. Please ensure it is in the directory.")
        exit()

    print("--- 3-BODY PRE-COMPUTATION MODULE START (Unequal Mass) ---")

    # Convert Solar Masses to SI
    m1_si = M1_SOLAR * M_SUN_SI
    m2_si = M2_SOLAR * M_SUN_SI
    M_binary_si = m1_si + m2_si

    r_max_si = calculate_initial_r_max(m1_si, m2_si, T_START)

    num_sims = len(M3_GRID_SOLAR) * len(R_RATIO_GRID) * len(D_RATIO_GRID)

    print(f"Inner Binary: m1={M1_SOLAR:.2e} M_sun, m2={M2_SOLAR:.2e} M_sun (Total: {M_binary_si/M_SUN_SI:.2e} M_sun)")
    print(f"Decoupling Radius (r_max): {r_max_si:.2e} meters")
    print(f"Running {num_sims} simulations...")

    results_log = []

    # Nested loops to cover the parameter space
    for m3_solar in tqdm(M3_GRID_SOLAR, desc="Perturber Mass (m3/M_sun)"):
        m3_si = m3_solar * M_SUN_SI

        for r_ratio in R_RATIO_GRID:
            r_initial = r_ratio * r_max_si

            for d_ratio in D_RATIO_GRID:
                d_perturber = d_ratio * r_initial

                # Run the N-body simulation
                run_single_three_body_sim(
                    r_initial, d_perturber, m1_si, m2_si, m3_si, M_binary_si, results_log
                )

    # --- Save Results ---
    output_filename = 'three_body_lookup_unequal.npz'

    # Convert list of dicts to dict of lists (NumPy format)
    if results_log:
        keys = results_log[0].keys()
        data = {key: np.array([d[key] for d in results_log], dtype=np.float64) for key in keys}

        # NOTE: For boolean/object arrays (like r_evolution), ensure proper handling.
        # NumPy requires all arrays in the savez to have the same length.
        # We save the raw float data.

        np.savez_compressed(output_filename, **data)

        print(f"\n✅ Pre-computation complete. {len(results_log)} results saved to '{output_filename}'")
    else:
        print("\n❌ Pre-computation failed. No results logged.")
