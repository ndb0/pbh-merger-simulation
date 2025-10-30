import torch
import math
import cosmo
import numpy as np

def generate_pbh_population_si(
    N: int,
    cfg: object, # We need the full config for f_pbh, mu, sigma
    seed: int,
    device: str,
    t_start_s: float # [s]
):
    """
    Generates the initial masses, positions (proper), and velocities (proper)
    for the PBH population *at t_start* for the Stage 1 sim.

    This function CALCULATES the correct box volume and includes initial
    velocity perturbations necessary for binary formation.

    !! CRITICAL: All units are SI (kg, m, s) !!
    """
    torch.manual_seed(seed)
    N = int(N) # [dimensionless]
    f_pbh = cfg.pbh_population.f_pbh # [dimensionless]

    # 1. Cosmological Context Calculation

    # Get initial time and scale factor
    a_start = cosmo.a_of_t(t_start_s) # [dimensionless]
    z_start = cosmo.z_of_a(a_start) # [dimensionless]

    # Get Dark Matter density at z_start
    rho_DM_start = cosmo.rho_DM_0 / (a_start**3) # [kg m^-3]

    # Get PBH density at z_start
    rho_PBH_start = f_pbh * rho_DM_start # [kg m^-3]

    # 2. Masses
    mu_kg = cfg.pbh_population.f_pbh_mu * cosmo.M_sun_SI # [kg]
    sigma = cfg.pbh_population.f_pbh_sigma # [dimensionless]
    log_mu = math.log(mu_kg) # [log(kg)]
    mass_dist = torch.distributions.LogNormal(log_mu, sigma)
    masses = mass_dist.sample((N, 1)).to(device)  # Shape (N, 1) # [kg]

    # Calculate the average mass of our sample to set the box size
    avg_mass_kg = torch.exp(torch.tensor(log_mu + (sigma**2 / 2.0))).item() # [kg]

    # Calculate the number density: n_PBH = rho_PBH / <m>
    n_PBH_start = rho_PBH_start / avg_mass_kg # [m^-3]

    # 3. Position and Box Size Calculation

    # Calculate the proper volume this box must have to match the density
    total_mass_in_box = N * avg_mass_kg # [kg]
    volume_m3 = total_mass_in_box / rho_PBH_start # [m^3]
    edge_len_m = volume_m3 ** (1./3.) # [m]

    # --- VERBOSE OUTPUT ---
    print(f"\n[SIM START (Seed {seed})]")
    print(f"  Simulating {N} particles.")
    print(f"  f_pbh (Fraction of DM): {f_pbh:.4e}")
    print(f"  Start Time (t_start): {t_start_s/cosmo.YEAR_S:.2e} yr (z={z_start:.1e})")
    print(f"  Avg PBH Mass: {avg_mass_kg/cosmo.M_sun_SI:.2f} M_sun")
    print(f"  PBH Mass Density (rho_PBH): {rho_PBH_start:.2e} kg/m^3")
    print(f"  PBH Number Density (n_PBH): {n_PBH_start:.2e} m^-3")
    print(f"  Total Box Mass: {total_mass_in_box/cosmo.M_sun_SI:.2e} M_sun")
    print(f"  Calculated Proper Box Size (L_m): {edge_len_m:.2e} m")
    print(f"    (L_pc): {edge_len_m/cosmo.parsec:.2e} pc")
    # ----------------------

    # Generate random positions within the proper volume
    positions = (torch.rand((N, 3), device=device) - 0.5) * edge_len_m # Shape (N, 3) # [m]

    # 4. Velocities (Hubble Flow + Peculiar Velocity Perturbation)

    # H_start = 0.5 / t_start_s in the radiation era
    H_start = 0.5 / t_start_s # H in s^-1

    # --- CRITICAL FIX: ADD PECULIAR VELOCITY (for angular momentum/tidal forces) ---

    # Hubble flow velocity component
    V_hubble = H_start * positions # Shape (N, 3) # [m s^-1]

    # Characteristic magnitude of Hubble velocity across the box
    V_hubble_mag = H_start * edge_len_m # [m/s]

    # We use a fractional multiplier (0.1% of V_hubble_mag scaled by f_pbh)
    # This introduces the necessary initial tidal torques/angular momentum.
    # We use a smaller base factor (0.001) to keep the perturbation tiny.
    V_peculiar_multiplier = 0.001 * f_pbh

    # Randomly sampled peculiar velocity vector
    V_peculiar = V_peculiar_multiplier * V_hubble_mag * torch.randn((N, 3), device=device) # [m s^-1]

    velocities = V_hubble + V_peculiar # Shape (N, 3) # [m s^-1]

    # ----------------------------------------------------------------------------

    return masses, positions, velocities


def generate_halo_population_si(
    N: int,
    mu_kg: float, # [kg]
    sigma: float, # [dimensionless]
    radius_m: float, # [m]
    seed: int,
    device: str,
    initial_binary_catalog: list,
    G: float # [m^3 kg^-1 s^-2]
):
    """
    Generates the initial state for a single Stage 2 halo simulation.
    """
    torch.manual_seed(seed)
    N = int(N) # [dimensionless]

    # --- 1. Populate single BHs ---
    log_mu = math.log(mu_kg) # [log(kg)]
    mass_dist = torch.distributions.LogNormal(log_mu, sigma)
    masses = mass_dist.sample((N, 1)).to(device) # Shape (N, 1) # [kg]

    # --- 2. Positions (Randomly in a sphere) ---
    r = radius_m * torch.pow(torch.rand((N, 1), device=device), 1./3.) # [m]
    theta = torch.acos(1.0 - 2.0 * torch.rand((N, 1), device=device)) # [rad]
    phi = 2.0 * np.pi * torch.rand((N, 1), device=device) # [rad]

    positions = torch.cat([
        r * torch.sin(theta) * torch.cos(phi),
        r * torch.sin(theta) * torch.sin(phi),
        r * torch.cos(theta)
    ], dim=1) # Shape (N, 3) # [m]

    # --- 3. Velocities (Virialized) ---
    M_total = torch.sum(masses) # [kg]
    v_disp_sq = 0.5 * G * M_total / radius_m # [m^2 s^-2]
    v_disp = torch.sqrt(v_disp_sq) # [m s^-1]

    # Sample from a Maxwell-Boltzmann distribution (approx by Gaussian)
    velocities = torch.randn((N, 3), device=device) * v_disp / torch.sqrt(torch.tensor(3.0)) # [m s^-1]

    # --- 4. Spins ---
    spins = torch.zeros((N, 3), device=device) # [dimensionless]

    # --- 5. Binary Info ---
    binary_info = {} # Empty for now

    return masses, positions, velocities, spins, binary_info

def update_simulation_state(positions, velocities, masses, spins, binary_info, updates):
    """
    Applies updates from the merger physics module to the main state tensors.
    """

    # 1. Get indices to keep
    remove_indices = sorted(list(set(updates['remove'])), reverse=True) # [int]
    N_old = masses.shape[0] # [int]
    keep_indices = [i for i in range(N_old) if i not in remove_indices] # [int]

    # 2. Create new tensors from kept particles
    positions_new = positions[keep_indices] # [m]
    velocities_new = velocities[keep_indices] # [m s^-1]
    masses_new = masses[keep_indices] # [kg]
    spins_new = spins[keep_indices] # [dimensionless]

    # 3. Create new binary_info
    binary_info_new = {}
    for new_idx, old_idx in enumerate(keep_indices):
        if old_idx in binary_info:
            binary_info_new[new_idx] = binary_info[old_idx]

    # 4. Add new particles from mergers
    new_particle_list = updates.get('add', [])
    if new_particle_list:
        add_pos = torch.stack([p['pos'] for p in new_particle_list]) # [m]
        add_vel = torch.stack([p['vel'] for p in new_particle_list]) # [m s^-1]
        add_mass = torch.stack([p['mass'] for p in new_particle_list]) # [kg]
        add_spin = torch.stack([p['spin'] for p in new_particle_list]) # [dimensionless]

        positions_new = torch.cat([positions_new, add_pos], dim=0)
        velocities_new = torch.cat([velocities_new, add_vel], dim=0)
        masses_new = torch.cat([masses_new, add_mass], dim=0)
        spins_new = torch.cat([spins_new, add_spin], dim=0)

        # Add new info for the remnants
        current_idx = len(keep_indices) # [int]
        for i, p in enumerate(new_particle_list):
            binary_info_new[current_idx + i] = {'generation': 2} # Placeholder

    return positions_new, velocities_new, masses_new, spins_new, binary_info_new
