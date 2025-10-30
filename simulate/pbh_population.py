import torch
import math
import cosmo
import numpy as np

def generate_pbh_population_si(
    N: int,
    mu_kg: float,
    sigma: float,
    volume_m3: float,
    seed: int,
    device: str,
    t_start_s: float
):
    """
    Generates the initial masses, positions (proper), and velocities (proper) 
    for the PBH population *at t_start* for the Stage 1 sim.
    
    !! CRITICAL: All units are SI (kg, m, s) !!
    """
    torch.manual_seed(seed)
    N = int(N)
    
    # 1. Masses
    # Convert mu_kg to log-normal mu
    log_mu = math.log(mu_kg) 
    mass_dist = torch.distributions.LogNormal(log_mu, sigma)
    masses = mass_dist.sample((N, 1)).to(device)  # Shape (N, 1) in kg
    
    # 2. Positions
    # Generate random positions within the proper volume
    edge_len_m = volume_m3 ** (1./3.)
    positions = (torch.rand((N, 3), device=device) - 0.5) * edge_len_m # Shape (N, 3) in m
    
    # 3. Velocities
    # Initial velocities are *not* zero. They have a "Hubble flow" component.
    # v = H(t) * r
    # H(t) = a_dot / a
    # In radiation era, a ~ t^(1/2), so a_dot = 0.5 * a / t
    # H(t) = 0.5 / t
    H_start = 0.5 / t_start_s # H in s^-1
    
    velocities = H_start * positions # Shape (N, 3) in m/s
    
    # TODO: Add peculiar velocities (random component)
    
    return masses, positions, velocities


def generate_halo_population_si(
    N: int,
    mu_kg: float,
    sigma: float,
    radius_m: float,
    seed: int,
    device: str,
    initial_binary_catalog: list,
    G: float
):
    """
    Generates the initial state for a single Stage 2 halo simulation.
    This includes a population of single BHs and "injects"
    binaries from the Stage 1 catalog.
    
    Returns:
        masses, positions, velocities, spins, binary_info (dict)
    """
    torch.manual_seed(seed)
    N = int(N)
    
    # --- 1. Populate single BHs ---
    log_mu = math.log(mu_kg)
    mass_dist = torch.distributions.LogNormal(log_mu, sigma)
    masses = mass_dist.sample((N, 1)).to(device) # Shape (N, 1) in kg
    
    # --- 2. TODO: Inject Binaries ---
    # This is a complex step. You would:
    # 1. Decide how many binaries to add (based on f_pbh, halo mass, etc.)
    # 2. Sample binaries from `initial_binary_catalog`.
    # 3. Remove 2*N_binaries from the `masses` tensor.
    # 4. Add N_binaries * 2 particles back, placing them in stable orbits
    #    (based on r_a, j) at random locations in the halo.
    # 5. Store their info in `binary_info` for tracking.
    
    # For now, we proceed with N single BHs.
    
    # 3. Positions
    # Randomly in a sphere
    r = radius_m * torch.pow(torch.rand((N, 1), device=device), 1./3.)
    theta = torch.acos(1.0 - 2.0 * torch.rand((N, 1), device=device))
    phi = 2.0 * np.pi * torch.rand((N, 1), device=device)
    
    positions = torch.cat([
        r * torch.sin(theta) * torch.cos(phi),
        r * torch.sin(theta) * torch.sin(phi),
        r * torch.cos(theta)
    ], dim=1) # Shape (N, 3) in m

    # 4. Velocities
    # We need a virialized velocity distribution.
    # V_esc^2 ~ G * M_total / R
    # V_disp^2 ~ V_esc^2 / 2 (approx)
    M_total = torch.sum(masses)
    v_disp_sq = 0.5 * G * M_total / radius_m
    v_disp = torch.sqrt(v_disp_sq)
    
    # Sample from a Maxwell-Boltzmann distribution (approx by Gaussian)
    velocities = torch.randn((N, 3), device=device) * v_disp / torch.sqrt(torch.tensor(3.0))

    # 5. Spins
    # Assume all primordial BHs are non-spinning
    spins = torch.zeros((N, 3), device=device)
    
    # 6. Binary Info
    # This dict tracks which particles are "special" (e.g., binaries, remnants)
    binary_info = {} # Empty for now
    
    return masses, positions, velocities, spins, binary_info

def update_simulation_state(positions, velocities, masses, spins, binary_info, updates):
    """
    Applies updates from the merger physics module to the main state tensors.
    This is complex because it changes the size of the tensors.
    """
    
    # 1. Get indices to keep
    remove_indices = sorted(list(set(updates['remove'])), reverse=True)
    N_old = masses.shape[0]
    keep_indices = [i for i in range(N_old) if i not in remove_indices]
    
    # 2. Create new tensors from kept particles
    positions_new = positions[keep_indices]
    velocities_new = velocities[keep_indices]
    masses_new = masses[keep_indices]
    spins_new = spins[keep_indices]
    
    # 3. Create new binary_info
    binary_info_new = {}
    for new_idx, old_idx in enumerate(keep_indices):
        if old_idx in binary_info:
            binary_info_new[new_idx] = binary_info[old_idx]
            
    # 4. Add new particles from mergers
    new_particle_list = updates.get('add', [])
    if new_particle_list:
        add_pos = torch.stack([p['pos'] for p in new_particle_list])
        add_vel = torch.stack([p['vel'] for p in new_particle_list])
        add_mass = torch.stack([p['mass'] for p in new_particle_list])
        add_spin = torch.stack([p['spin'] for p in new_particle_list])
        
        positions_new = torch.cat([positions_new, add_pos], dim=0)
        velocities_new = torch.cat([velocities_new, add_vel], dim=0)
        masses_new = torch.cat([masses_new, add_mass], dim=0)
        spins_new = torch.cat([spins_new, add_spin], dim=0)
        
        # Add new info for the remnants
        current_idx = len(keep_indices)
        for i, p in enumerate(new_particle_list):
            binary_info_new[current_idx + i] = {'generation': 2} # Placeholder
            
    return positions_new, velocities_new, masses_new, spins_new, binary_info_new

