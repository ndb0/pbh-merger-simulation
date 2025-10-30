import torch
import numpy as np
import cosmo

def calculate_coalescence_time(r_a, j, m1, m2, G, c):
    """
    Calculates the coalescence time (tau) for a binary (Peters' formula).
    """

    M = m1 + m2 # [kg]
    eta = (m1 * m2) / (M**2) # [dimensionless]

    if isinstance(j, torch.Tensor):
        large_time = 1e50 # Effectively infinity
        safe_j = torch.clamp(j, min=1e-15) # Prevent division by zero
        tau = torch.where(j < 1e-9, torch.tensor(large_time, device=j.device),
                          (3.0 * (c**5)) / (85.0 * (G**3)) * (r_a**4 * safe_j**7) / (eta * M**3)) # [s]
    else: # Handle float case
        if j < 1e-9:
             return 1e50 # Effectively infinity # [s]
        const_factor = (3.0 * (c**5)) / (85.0 * (G**3)) # [kg^3 s m^-4]
        tau = const_factor * (r_a**4 * j**7) / (eta * M**3) # [s]

    return tau


def calculate_remnant_properties(m1, m2, s1_vec, s2_vec):
    """
    Calculates the mass, spin, and recoil kick of the merger remnant (Simplified).
    """

    # 1. Remnant Mass (approximate)
    m_remnant = m1 + m2 # [kg]

    # 2. Remnant Spin (approximate)
    s_remnant_mag = 0.7 # Placeholder for spin magnitude
    s_remnant = torch.tensor([0.0, 0.0, s_remnant_mag], device=m1.device) # [dimensionless]

    # 3. Recoil Kick (Simplified Mass-Ratio Kick)
    q = m1 / m2 if m1 < m2 else m2 / m1 # mass ratio q <= 1 [dimensionless]
    eta = q / ((1 + q)**2) # [dimensionless]
    v_max_kick = 175e3 # 175 km/s [m s^-1]

    # Placeholder for Mass-Ratio Recoil
    v_kick_mag = v_max_kick * torch.sin((eta / 0.25) * np.pi) # [m s^-1]

    # Kick direction is random 3D (Simplification)
    v_recoil_dir = torch.randn(3, device=m1.device) # [dimensionless]
    v_recoil_dir = v_recoil_dir / torch.norm(v_recoil_dir) # [dimensionless]

    v_recoil = v_recoil_dir * v_kick_mag # [m s^-1]

    return m_remnant, s_remnant, v_recoil

def detect_and_handle_events(positions, velocities, masses, spins, binary_info, dt_s, t_s, G, c):
    """
    Scans all particles to find mergers and disruptions in this timestep.
    """

    N = masses.shape[0] # [int]
    if N < 2:
        return [], [], {'remove': [], 'add': []}

    new_mergers = []
    new_disruptions = []

    state_updates = {'remove': [], 'add': []}

    processed_indices = set()

    # --- 1. Check for Hierarchical Mergers ---
    for i in range(N):
        if i in processed_indices: continue

        for j in range(i + 1, N):
            if j in processed_indices: continue

            m1, m2 = masses[i, 0], masses[j, 0] # [kg]
            M_total = m1 + m2 # [kg]

            r_vec = positions[j] - positions[i] # [m]
            r_norm = torch.norm(r_vec) # [m]

            v_vec = velocities[j] - velocities[i] # [m s^-1]
            v_norm_sq = torch.dot(v_vec, v_vec) # [m^2 s^-2]

            # Check if they are bound (E_total < 0)
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

                # Calculate j
                j_denom = mu * torch.sqrt(G * M_total * r_a) # [kg m^2 s^-1]
                j = L_norm / j_denom # [dimensionless]
                j = torch.clamp(j, min=0.0, max=1.0)

                tau_s = calculate_coalescence_time(r_a, j, m1, m2, G, c) # [s]

                # --- MERGER ---
                if tau_s <= dt_s:
                    # This binary merges *this timestep*!
                    m_rem, s_rem, v_recoil = calculate_remnant_properties(
                        m1, m2, spins[i], spins[j]
                    )

                    # Calculate remnant position and velocity (CoM)
                    pos_rem = (positions[i] * m1 + positions[j] * m2) / M_total # [m]
                    vel_rem = (velocities[i] * m1 + velocities[j] * m2) / M_total # [m s^-1]
                    vel_rem = vel_rem + v_recoil # Add the kick! # [m s^-1]

                    # Log the merger
                    gen = 1
                    if i in binary_info and 'generation' in binary_info[i]: gen = max(gen, binary_info[i]['generation'])
                    if j in binary_info and 'generation' in binary_info[j]: gen = max(gen, binary_info[j]['generation'])

                    new_mergers.append({
                        't_s': t_s + tau_s, 'm1_kg': m1.item(), 'm2_kg': m2.item(),
                        'm_rem_kg': m_rem.item(), 'generation': gen + 1
                    })

                    # Schedule state updates
                    state_updates['remove'].extend([i, j])
                    state_updates['add'].append({
                        'pos': pos_rem, 'vel': vel_rem, 'mass': m_rem.view(1), 'spin': s_rem
                    })

                    processed_indices.add(i)
                    processed_indices.add(j)
                    break

    # --- 2. Check for Disruptions (Not implemented here) ---

    return new_mergers, new_disruptions, state_updates
