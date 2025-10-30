import torch
import numpy as np
import cosmo

def calculate_coalescence_time(r_a, j, m1, m2, G, c):
    """
    Calculates the coalescence time (tau) for a binary.
    This is based on Peters' formula (Eq. 7 in the paper).
    
    Args:
        r_a (torch.Tensor or float): Semi-major axis in meters
        j (torch.Tensor or float): Dimensionless angular momentum
        m1, m2 (torch.Tensor or float): Masses in kg
        G (float): Gravitational constant (SI)
        c (float): Speed of light (SI)
        
    Returns:
        torch.Tensor or float: Coalescence time in seconds
    """
    M = m1 + m2
    eta = (m1 * m2) / (M**2)
    
    # Convert to geometric units for the formula
    # r_a_geom = r_a_m * (G / c^2)  -- No, r_a is already a length
    # M_geom = M_kg * G / c^2
    # t_geom = t_s * c
    
    # Let's use the full SI formula to avoid unit errors.
    # tau = (3/85) * (r_a^4 * j^7) / (eta * M^3)  <- This is in GEOMETRIC units
    
    # Full SI conversion of Peters' formula:
    # tau = (5 / 256) * (c^5 / G^3) * (r_a^4 / (M^2 * mu))
    # This is for circular (j=1). The j dependence is (3/85) * ... * j^7
    # Let's find the ratio: ( (3/85) * j^7 ) / ( (5/256) * 1^7 )
    # No, that's not right.
    
    # Let's re-implement Eq. (7) with units:
    # tau (seconds) = tau (geometric) / c
    # r_a (geometric) = r_a (meters)
    # M (geometric) = M (kg) * G / c^2
    # eta is dimensionless
    
    # tau_geom = (3 / 85) * (r_a**4 * j**7) / (eta * M_geom**3)
    # tau_s = tau_geom / c
    # tau_s = (3 / (85 * c)) * (r_a**4 * j**7) / (eta * (M * G / c**2)**3)
    # tau_s = (3 / (85 * c)) * (r_a**4 * j**7) / (eta * M**3 * G**3 / c**6)
    # tau_s = (3 * c**5) / (85 * G**3) * (r_a**4 * j**7) / (eta * M**3)
    
    # This seems dimensionally correct:
    # (m/s)^5 / (m^3/kg/s^2)^3 = (m^5/s^5) / (m^9/kg^3/s^6) = (m^5 s^6 kg^3) / (s^5 m^9) = kg^3 s / m^4
    # (m^4) / (kg^3)
    # (kg^3 s / m^4) * (m^4 / kg^3) = s (seconds). Correct.
    
    if j < 1e-9: # Avoid division by zero for head-on collisions
        return 0.0 
        
    const_factor = (3.0 * (c**5)) / (85.0 * (G**3))
    tau = const_factor * (r_a**4 * j**7) / (eta * M**3)
    return tau

def calculate_remnant_properties(m1, m2, s1_vec, s2_vec):
    """
    Calculates the mass, spin, and recoil kick of the merger remnant.
    This uses phenomenological fits from numerical relativity.
    
    NOTE: This is a simplified placeholder. A real implementation
    requires complex fitting formulas (e.g., from Campanelli, Lousto, etc.)
    
    Args:
        m1, m2 (torch.Tensor): Masses in kg
        s1_vec, s2_vec (torch.Tensor): Dimensionless spin vectors (shape 3)
        
    Returns:
        (float, torch.Tensor, torch.Tensor): 
        m_remnant (kg), s_remnant (shape 3), v_recoil (shape 3, m/s)
    """
    
    # 1. Remnant Mass (approximate, ignores radiated energy)
    m_remnant = m1 + m2 
    
    # 2. Remnant Spin (approximate)
    # For primordial BHs, we assume initial spins (s1, s2) are zero.
    # The remnant spin comes from the orbital angular momentum.
    # A_final / M_final^2 ≈ 0.68 (from GW150914)
    # This is a placeholder. Real calculation is complex.
    s_remnant = torch.tensor([0.0, 0.0, 0.7], device=m1.device)
    
    # 3. Recoil Kick
    # The GW recoil kick is highly dependent on mass ratios and spins.
    # For non-spinning binaries, the kick is non-zero if masses are unequal.
    # Max kick ~175 km/s for q ~ 0.36
    # For spinning binaries, kicks can be up to 5000 km/s.
    
    # Since we assume s1=s2=0, we only have the mass-ratio kick.
    # This is a simple fit (placeholder):
    q = m1 / m2 if m1 < m2 else m2 / m1 # mass ratio q <= 1
    eta = q / ((1 + q)**2)
    
    # Max kick at eta=0.19 (q=0.36). Kick is 0 at eta=0 and eta=0.25 (q=1).
    # Fit with a simple sine wave
    v_max_kick = 175e3 # 175 km/s
    # Map eta from [0, 0.25] to [0, pi]
    v_kick_mag = v_max_kick * torch.sin((eta / 0.25) * np.pi)
    
    # Kick direction is random in the orbital plane.
    # We don't know the orbital plane, so we'll pick a random 3D direction.
    # A better sim would use the L_vec to define the plane.
    v_recoil_dir = torch.randn(3, device=m1.device)
    v_recoil_dir = v_recoil_dir / torch.norm(v_recoil_dir)
    
    v_recoil = v_recoil_dir * v_kick_mag
    
    return m_remnant, s_remnant, v_recoil

def detect_and_handle_events(positions, velocities, masses, spins, binary_info, dt_s, t_s, G, c):
    """
    Scans all particles to find mergers and disruptions in this timestep.
    This is the core of the Stage 2 physics.
    """
    
    N = masses.shape[0]
    if N < 2:
        return [], [], []
        
    new_mergers = []
    new_disruptions = []
    
    # List of updates to apply to the state
    # 'remove': [idx1, idx2], 'add': [pos, vel, mass, spin]
    state_updates = {'remove': [], 'add': []}
    
    # Keep track of indices we've already dealt with
    processed_indices = set()
    
    # --- 1. Check for Hierarchical Mergers ---
    # We check all unique pairs
    for i in range(N):
        if i in processed_indices: continue
            
        for j in range(i + 1, N):
            if j in processed_indices: continue

            # Calculate orbital parameters
            m1, m2 = masses[i, 0], masses[j, 0]
            M_total = m1 + m2
            
            r_vec = positions[j] - positions[i]
            r_norm = torch.norm(r_vec)
            
            v_vec = velocities[j] - velocities[i]
            v_norm_sq = torch.dot(v_vec, v_vec)
            
            # Check if they are bound
            if 0.5 * v_norm_sq < (G * M_total / r_norm):
                # They are bound. Calculate coalescence time.
                mu = (m1 * m2) / M_total
                L_vec = mu * torch.cross(r_vec, v_vec)
                L_norm = torch.norm(L_vec)
                
                r_a = 1.0 / ( (2.0 / r_norm) - (v_norm_sq / (G * M_total)) )
                j = (L_norm / mu) / torch.sqrt(r_a * M_total * G)

                tau_s = calculate_coalescence_time(r_a, j, m1, m2, G, c)
                
                # --- MERGER ---
                if tau_s <= dt_s:
                    # This binary merges *this timestep*!
                    # Get remnant properties
                    m_rem, s_rem, v_recoil = calculate_remnant_properties(
                        m1, m2, spins[i], spins[j]
                    )
                    
                    # Calculate remnant position and velocity (CoM)
                    pos_rem = (positions[i] * m1 + positions[j] * m2) / M_total
                    vel_rem = (velocities[i] * m1 + velocities[j] * m2) / M_total
                    vel_rem = vel_rem + v_recoil # Add the kick!
                    
                    # Log the merger
                    gen = 1
                    if i in binary_info and 'generation' in binary_info[i]:
                        gen = max(gen, binary_info[i]['generation'])
                    if j in binary_info and 'generation' in binary_info[j]:
                        gen = max(gen, binary_info[j]['generation'])
                    
                    new_mergers.append({
                        't_s': t_s + tau_s,
                        'm1_kg': m1.item(),
                        'm2_kg': m2.item(),
                        'm_rem_kg': m_rem.item(),
                        'generation': gen if (i in binary_info or j in binary_info) else (binary_info.get(i,{}).get('generation', 1) + binary_info.get(j,{}).get('generation', 1))
                    })
                    
                    # Schedule state updates
                    state_updates['remove'].extend([i, j])
                    state_updates['add'].append({
                        'pos': pos_rem, 'vel': vel_rem, 'mass': m_rem.view(1), 'spin': s_rem
                    })
                    
                    processed_indices.add(i)
                    processed_indices.add(j)
                    break # Move to the next 'i'
                    
    # --- 2. Check for Disruptions (S_L) ---
    # This is more complex. A disruption happens when a 3rd body
    # interacts with a binary and makes it unbound.
    # We need to track the "initial" binaries from Stage 1.
    
    # (Simplified: This part is complex and requires tracking binary state
    # across timesteps. A full implementation is a major task.)
    
    # We also log late 2-body and 3-body captures here, which are
    # simply new binaries formed with tau_s > dt_s.
    
    return new_mergers, new_disruptions, state_updates

