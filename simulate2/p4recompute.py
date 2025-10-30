import torch
import numpy as np
import cosmo
import nbody_kernels # Assumed to contain get_acceleration_early_universe and leapfrog
import os
from tqdm import tqdm
from math import log10
import math # Ensure math is imported for math.sqrt

# --- Configuration for Pre-computation ---

# FIXED MASSES for the Inner Binary (m1 and m2) in Solar Masses (M_sun)
M1_SOLAR = 1.0
M2_SOLAR = 1e-3
M_BINARY_SOLAR = M1_SOLAR + M2_SOLAR

# PERTURBER MASS GRID (m3) - From 1e-3 M_sun to 1 M_sun
M3_GRID_SOLAR = np.geomspace(1e-3, 1.0, 6) # 6 steps for better resolution

# SPATIAL GRIDS (Dimensionless ratios)
# Vary r_initial from r_max / 1e10 to r_max
R_RATIO_GRID = np.geomspace(1e-10, 1.0, 10) # 10 steps: Ranging from 10^-10 to 1.0 of r_max
D_RATIO_GRID = np.linspace(1.5, 5.0, 5) # d_perturber / r_initial

# SIMULATION PARAMETERS
NUM_TIMESTEPS = 2500
Z_START = 1e9
T_START = cosmo.t_of_a(cosmo.a_of_z(Z_START))
T_END = cosmo.t_eq
TIME_GRID = np.geomspace(T_START, T_END, NUM_TIMESTEPS)

# --- Fixed Cosmological and Physical Parameters ---
G_SI = cosmo.G
M_SUN_SI = cosmo.M_sun_SI

# --- OVERRIDE KERNEL FOR PRE-COMPUTATION ---
# This wrapper overrides the acceleration function to remove the Hubble drag term,
# forcing the simulation to behave as a purely gravitational system (in a comoving patch).
def get_acceleration_pure_grav(positions, masses, t, G, a_eq, t_eq):
     # NOTE: t, a_eq, t_eq are ignored, as this is for pure gravity.
     return nbody_kernels.get_acceleration_late_universe(positions, masses, G)

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
    Sets up the initial state for the 3-body system at t_start with ZERO relative velocity.
    """

    masses = torch.tensor([[m1_si], [m2_si], [m3_si]], dtype=torch.float64) # [kg]
    M_inner = m1_si + m2_si

    # 1. Positions (Physical coordinates, CoM at origin)
    r1_rel = -(m2_si / M_inner) * r_initial
    r2_rel = (m1_si / M_inner) * r_initial

    pos1 = torch.tensor([r1_rel, 0.0, 0.0]) # [m]
    pos2 = torch.tensor([r2_rel, 0.0, 0.0]) # [m]
    pos3 = torch.tensor([0.0, d_perturber, 0.0]) # [m]

    positions = torch.stack([pos1, pos2, pos3]) # [m]

    # 2. Velocities (Zero Relative Velocity)
    # --- CRITICAL FIX: SET ALL INITIAL VELOCITIES TO ZERO ---
    # This ensures the system starts with maximum gravitational potential energy
    # dominance, allowing stable orbits to form immediately if r_initial is small enough.
    velocities = torch.zeros((3, 3), dtype=torch.float64) # [m/s]

    # Final CoM subtraction for the 3-body system (CoM velocity must be zero)
    M_total_system = m1_si + m2_si + m3_si
    # Need to handle potential division by zero if M_total_system is zero
    if M_total_system > 1e-9: # Use a small threshold
        V_com = torch.sum(masses * velocities.unsqueeze(1), dim=0) / M_total_system
        velocities = velocities - V_com # [m/s]
    # else: velocities remain zero

    return masses, positions, velocities

def run_single_three_body_sim(r_initial, d_perturber, m1_si, m2_si, m3_si, M_binary_si, log_buffer):
    """
    Runs one simulation and records the outcome (survival, disruption, merger).
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

        # --- CRITICAL CHANGE: Use the PURE GRAVITY kernel override ---
        positions, velocities = nbody_kernels.leapfrog_kick_drift_kick(
            positions, velocities, masses,
            get_acceleration_pure_grav, # <- Using the PURE GRAVITY wrapper
            dt, t, **kernel_args
        )
        # -------------------------------------------------------------

    # --- ANALYSIS OF FINAL STATE (t = t_eq) ---

    # Check the state of the inner binary (particles 0 and 1)
    # Check if masses tensor has expected dimensions
    if masses.shape[0] < 2:
        print(f"Warning: Less than 2 particles remaining at end of sim (r_init={r_initial/r_max:.2e}, d_pert={d_perturber/r_initial:.2e}, q_pert={m3_si/M_binary_si:.2e}). Skipping analysis.")
        is_bound = False
        is_merged = False # Cannot merge if particles missing
        r_a_final = 0.0
        j_final = 0.0
    else:
        m1, m2 = masses[0, 0], masses[1, 0] # Ensure indexing is correct

        r_vec = positions[1] - positions[0]
        r_norm = torch.norm(r_vec)
        v_vec = velocities[1] - velocities[0]

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
            # Add check for E_total being too close to zero
            if torch.abs(E_total) < 1e-15 * torch.abs(E_pot): # Relative check
                r_a_final = float('inf') # Mark as unbound/parabolic
                is_bound = False # Correct status
            else:
                 r_a_final = -G_SI * m1 * m2 / (2.0 * E_total)

            # Only proceed if r_a is valid
            if torch.isfinite(r_a_final) and r_a_final > 0:
                L_vec = mu * torch.cross(r_vec, v_vec)
                L_norm = torch.norm(L_vec)
                j_denom = mu * torch.sqrt(G_SI * M_binary_si * r_a_final)
                # Check for zero denominator
                if j_denom < 1e-15 * L_norm: # Relative check
                    j_final = 0.0
                else:
                    j_final = (L_norm / j_denom).item()

                # Calculate Coalescence Time (tau)
                tau_s = merger_physics.calculate_coalescence_time(r_a_final, j_final, m1, m2, G_SI, cosmo.c)

                # Check for merger (tau < age of universe)
                if tau_s < cosmo.T_0_S:
                    is_merged = True
            else: # If r_a is inf or zero, it's not truly bound in an orbital sense
                is_bound = False
                r_a_final = 0.0 # Reset for logging

    # Record result to the buffer
    log_buffer.append({
        'r_initial': r_initial / r_max,
        'd_perturber': d_perturber / r_initial,
        'q_perturber': m3_si / M_binary_si,
        'is_bound': is_bound,
        'is_merged': is_merged,
        'r_a_final': r_a_final / r_max if is_bound and torch.isfinite(r_a_final) else 0.0, # Use r_max scaling only if bound
        'j_final': j_final,
        # Log the full evolution data for visualization
        'r_evolution': r_evolution,
        'time_grid': TIME_GRID
    })

    return is_bound, is_merged

if __name__ == "__main__":

    try:
        from merger_physics import calculate_coalescence_time
        import merger_physics
    except ImportError:
        print("ERROR: Could not import 'merger_physics.py'. Please ensure it is in the directory.")
        exit()

    print("--- 3-BODY PRE-COMPUTATION MODULE START (Final Attempt) ---")

    # Convert Solar Masses to SI
    m1_si = M1_SOLAR * M_SUN_SI
    m2_si = M2_SOLAR * M_SUN_SI
    M_binary_si = m1_si + m2_si

    r_max_si = calculate_initial_r_max(m1_si, m2_si, T_START)

    num_sims = len(M3_GRID_SOLAR) * len(R_RATIO_GRID) * len(D_RATIO_GRID)

    print(f"Inner Binary: m1={M1_SOLAR:.2e} M_sun, m2={M2_SOLAR:.2e} M_sun (Total: {M_binary_si/M_SUN_SI:.2e} M_sun)")
    print(f"Decoupling Radius (r_max): {r_max_si:.2e} meters")
    print(f"Running {num_sims} simulations...")
    print("NOTE: Using PURE GRAVITY kernel (Hubble acceleration removed) and ZERO initial relative velocity.")

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
    output_filename = 'three_body_lookup_unequal_final.npz'

    # Convert list of dicts to dict of lists (NumPy format)
    if results_log:
        keys = results_log[0].keys()
        # Use dtype=object for r_evolution and time_grid since they are long arrays
        data = {}
        for key in keys:
            if key in ['r_evolution', 'time_grid']:
                data[key] = np.array([d[key] for d in results_log], dtype=object)
            else:
                 # Check for NaN/Inf before converting to float64
                 temp_array = [d[key] for d in results_log]
                 if any(isinstance(x, bool) for x in temp_array): # Handle booleans separately
                     data[key] = np.array(temp_array, dtype=bool)
                 else:
                     # Replace potential non-finite values before conversion
                     clean_array = [x if np.isfinite(x) else 0.0 for x in temp_array]
                     data[key] = np.array(clean_array, dtype=np.float64)

        np.savez_compressed(output_filename, **data)

        print(f"\n✅ Pre-computation complete. {len(results_log)} results saved to '{output_filename}'")
    else:
        print("\n❌ Pre-computation failed. No results logged.")
