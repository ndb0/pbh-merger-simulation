import torch
import numpy as np
import cosmo
import nbody_kernels # Assumed to contain get_acceleration_early_universe and leapfrog
import os
from tqdm import tqdm
from math import log10
import math # Ensure math is imported for math.sqrt

# --- Configuration for Pre-computation ---
M1_SOLAR = 1.0
M2_SOLAR = 1e-3
M_BINARY_SOLAR = M1_SOLAR + M2_SOLAR
M3_GRID_SOLAR = np.geomspace(1e-3, 1.0, 6)
R_RATIO_GRID = np.geomspace(1e-10, 1.0, 10) # User-requested range
D_RATIO_GRID = np.linspace(1.5, 5.0, 5)

# --- Simulation Time Parameters ---
Z_START = 1e9
T_START_SI = cosmo.t_of_a(cosmo.a_of_z(Z_START))
T_END_SI = cosmo.t_eq

# --- CRITICAL FIX: Two-Stage Time Grid ---
# Phase A: High-resolution capture of the initial orbit
T_PHASE_A_END_SI = T_START_SI + 30.0 # Run high-res for 30 seconds
NUM_TIMESTEPS_A = 30000 # 0.001s resolution (30s / 0.001s)
TIME_GRID_A = np.linspace(T_START_SI, T_PHASE_A_END_SI, NUM_TIMESTEPS_A)

# Phase B: Logarithmic grid for cosmic expansion
NUM_TIMESTEPS_B = 2500
TIME_GRID_B = np.geomspace(T_PHASE_A_END_SI, T_END_SI, NUM_TIMESTEPS_B)

# Combine the grids
TIME_GRID = np.concatenate((TIME_GRID_A, TIME_GRID_B))
NUM_TIMESTEPS = len(TIME_GRID)
# ---

# --- Fixed Cosmological and Physical Parameters ---
G_SI = cosmo.G
M_SUN_SI = cosmo.M_sun_SI

# --- Use FULL Cosmological Kernel ---
def get_acceleration_cosmological(positions, masses, t, G, a_eq, t_eq):
     return nbody_kernels.get_acceleration_early_universe(positions, masses, t, G=G, a_eq=a_eq, t_eq=t_eq)

def calculate_initial_r_max(m1_si, m2_si, t_start):
    """
    Calculates the characteristic physical separation (r_max).
    Corrected formula: r_max ~ (2 G M / H^2)^(1/3).
    """
    M_total = m1_si + m2_si # [kg]
    H_start = 0.5 / t_start # [s^-1]
    H_sq = H_start**2 # [s^-2]
    r_max_cubed = (2 * G_SI * M_total) / H_sq # [m^3]
    r_max = r_max_cubed**(1.0/3.0) # [m]
    return r_max # [m]

def setup_three_body_si(r_initial, d_perturber, m1_si, m2_si, m3_si, t_start):
    """
    Sets up the initial state for the 3-body system at t_start using Hubble flow
    and a stabilizing angular kick.
    """

    masses = torch.tensor([[m1_si], [m2_si], [m3_si]], dtype=torch.float64) # [kg]
    M_inner = m1_si + m2_si

    # 1. Positions
    r1_rel = -(m2_si / M_inner) * r_initial
    r2_rel = (m1_si / M_inner) * r_initial
    pos1 = torch.tensor([r1_rel, 0.0, 0.0]) # [m]
    pos2 = torch.tensor([r2_rel, 0.0, 0.0]) # [m]
    pos3 = torch.tensor([0.0, d_perturber, 0.0]) # [m]
    positions = torch.stack([pos1, pos2, pos3]) # [m]

    # 2. Velocities (Hubble Flow + Stabilizing Kick)
    H_start = 0.5 / t_start # [s^-1]
    V_hubble = H_start * positions # [m/s]

    # Stabilizing Kick
    if r_initial < 1e-15: V_kepler = 0.0
    else:
        V_kepler_sq = (G_SI * M_inner) / r_initial
        V_kepler = math.sqrt(V_kepler_sq) if V_kepler_sq > 0 else 0.0 # [m/s]

    V_tangential_scale = 1e-6 * V_kepler # [m/s]
    V1_rel_mag = (m2_si / M_inner) * V_tangential_scale
    V2_rel_mag = (m1_si / M_inner) * V_tangential_scale

    V_peculiar = torch.zeros((3, 3), dtype=torch.float64)
    V_peculiar[0, 1] = V1_rel_mag
    V_peculiar[1, 1] = -V2_rel_mag

    velocities = V_hubble + V_peculiar # [m/s]

    # Final CoM subtraction
    M_total_system = m1_si + m2_si + m3_si
    if M_total_system > 1e-15:
        V_com = torch.sum(masses * velocities.unsqueeze(1), dim=0) / M_total_system
        velocities = velocities - V_com # [m/s]

    return masses, positions, velocities

def run_single_three_body_sim(r_initial, d_perturber, m1_si, m2_si, m3_si, M_binary_si, log_buffer):
    """
    Runs one simulation and records the outcome (survival, disruption, merger).
    Uses the full cosmological kernel.
    """

    r_max = calculate_initial_r_max(m1_si, m2_si, T_START_SI) # r_max for the inner binary

    # Array to log orbital separation |r1 - r2| at each step for plotting
    r_evolution = np.zeros(NUM_TIMESTEPS)

    # Set up initial conditions
    masses, positions, velocities = setup_three_body_si(
        r_initial, d_perturber, m1_si, m2_si, m3_si, T_START_SI
    )

    # N-body simulation loop
    kernel_args = {'G': G_SI, 'a_eq': cosmo.a_eq, 't_eq': cosmo.t_eq}

    for k in range(NUM_TIMESTEPS - 1): # Loop over the combined time grid

        # 1. Log Separation
        if positions.shape[0] >= 2:
            r_vec_log = positions[1] - positions[0]
            r_evolution[k] = torch.norm(r_vec_log).item()
        else: r_evolution[k] = np.nan

        t = TIME_GRID[k]
        dt = TIME_GRID[k+1] - t
        if dt <= 0: continue

        # --- Using the FULL COSMOLOGICAL KERNEL ---
        positions, velocities = nbody_kernels.leapfrog_kick_drift_kick(
            positions, velocities, masses,
            nbody_kernels.get_acceleration_early_universe, # <--- FULL KERNEL
            dt, t, **kernel_args
        )
        # ---------------------------------------------

    # Log the final step's position
    if positions.shape[0] >= 2:
        r_evolution[-1] = torch.norm(positions[1] - positions[0]).item()
    else:
        r_evolution[-1] = np.nan

    # --- ANALYSIS OF FINAL STATE (t = t_eq) ---
    is_bound = False
    is_merged = False
    r_a_final = 0.0
    j_final = 0.0

    if masses.shape[0] >= 2 and positions.shape[0] >= 2 and velocities.shape[0] >= 2:
        m1, m2 = masses[0, 0], masses[1, 0]
        r_vec = positions[1] - positions[0]
        r_norm = torch.norm(r_vec)
        v_vec = velocities[1] - velocities[0]
        v_norm_sq = torch.dot(v_vec, v_vec)

        if r_norm < 1e-15:
            is_merged = True
        else:
            mu = (m1 * m2) / M_binary_si
            E_kin = 0.5 * mu * v_norm_sq
            E_pot = -G_SI * m1 * m2 / r_norm
            E_total = E_kin + E_pot
            is_bound = E_total < 0

            if is_bound:
                if torch.abs(E_total) < 1e-15 * torch.abs(E_pot):
                    is_bound = False
                else:
                     r_a_final = -G_SI * m1 * m2 / (2.0 * E_total)

                if torch.isfinite(r_a_final) and r_a_final > 0:
                    L_vec = mu * torch.cross(r_vec, v_vec)
                    L_norm = torch.norm(L_vec)
                    if r_a_final > 1e-15:
                         j_denom = mu * torch.sqrt(G_SI * M_binary_si * r_a_final)
                         if j_denom < 1e-15 * L_norm:
                             j_final = 0.0
                         else:
                             j_final = (L_norm / j_denom).item()

                         tau_s = merger_physics.calculate_coalescence_time(r_a_final, j_final, m1, m2, G_SI, cosmo.c)
                         if tau_s < cosmo.T_0_S: is_merged = True
                    else:
                        is_bound = False; is_merged = True; j_final = 0.0
                else:
                    is_bound = False; r_a_final = 0.0

    # Record result
    log_buffer.append({
        'r_initial': r_initial / r_max, 'd_perturber': d_perturber / r_initial,
        'q_perturber': m3_si / M_binary_si, 'is_bound': is_bound, 'is_merged': is_merged,
        'r_a_final': r_a_final / r_max if is_bound and torch.isfinite(r_a_final) and r_a_final > 0 else 0.0,
        'j_final': j_final, 'r_evolution': r_evolution, 'time_grid': TIME_GRID
    })

    return is_bound, is_merged

if __name__ == "__main__":

    try:
        from merger_physics import calculate_coalescence_time
        import merger_physics
    except ImportError:
        print("ERROR: Could not import 'merger_physics.py'.")
        exit()

    print(f"--- 3-BODY PRE-COMPUTATION MODULE START (High-Res Timestep) ---")

    # Convert Solar Masses to SI
    m1_si = M1_SOLAR * M_SUN_SI
    m2_si = M2_SOLAR * M_SUN_SI
    M_binary_si = m1_si + m2_si

    r_max_si = calculate_initial_r_max(m1_si, m2_si, T_START_SI)

    num_sims = len(M3_GRID_SOLAR) * len(R_RATIO_GRID) * len(D_RATIO_GRID)

    print(f"Inner Binary: m1={M1_SOLAR:.2e} M_sun, m2={M2_SOLAR:.2e} M_sun (Total: {M_binary_si/M_SUN_SI:.2e} M_sun)")
    print(f"Corrected Decoupling Radius (r_max): {r_max_si:.2e} meters")
    print(f"Running {num_sims} simulations with NUM_TIMESTEPS = {NUM_TIMESTEPS}...")
    print("NOTE: Using FULL cosmological kernel with 2-stage timestep (High-res capture + Log expansion).")

    results_log = []

    # Nested loops
    for m3_solar in tqdm(M3_GRID_SOLAR, desc="Perturber Mass (m3/M_sun)"):
        m3_si = m3_solar * M_SUN_SI
        for r_ratio in R_RATIO_GRID:
            r_initial = r_ratio * r_max_si
            for d_ratio in D_RATIO_GRID:
                d_perturber = d_ratio * r_initial
                run_single_three_body_sim(
                    r_initial, d_perturber, m1_si, m2_si, m3_si, M_binary_si, results_log
                )

    # Save Results
    output_filename = 'three_body_lookup_unequal_final.npz'
    if results_log:
        keys = results_log[0].keys()
        data = {}
        for key in keys:
            if key in ['r_evolution', 'time_grid']:
                data[key] = np.array([d[key] for d in results_log], dtype=object)
            else:
                 temp_array = [d[key] for d in results_log]
                 if key in ['is_bound', 'is_merged']:
                     data[key] = np.array(temp_array, dtype=bool)
                 else:
                     clean_array = [x if np.isfinite(x) else 0.0 for x in temp_array]
                     data[key] = np.array(clean_array, dtype=np.float64)

        np.savez_compressed(output_filename, **data)
        print(f"\n✅ Pre-computation complete. {len(results_log)} results saved to '{output_filename}'")
    else:
        print("\n❌ Pre-computation failed.")
