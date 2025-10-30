import torch
import cosmo

def get_acceleration_early_universe(positions, masses, t, G, a_eq, t_eq):
    """
    Calculates the acceleration on N bodies in the EARLY (radiation-dominated) universe.
    
    This includes two forces:
    1. Standard Newtonian Gravity.
    2. The "Hubble Drag" force from the cosmic expansion, F = m * (a_ddot/a) * r
       (from Eq. 18 in the paper).
       
    The simulation runs in PROPER coordinates (in meters), not comoving.
    
    Args:
        positions (torch.Tensor): Shape (N, 3) - Proper positions in meters. # [m]
        masses (torch.Tensor): Shape (N, 1) - Masses in kg. # [kg]
        t (float): Current time in seconds. # [s]
        G (float): Gravitational constant in SI units. # [m^3 kg^-1 s^-2]
        a_eq (float): Scale factor at matter-radiation equality. # [dimensionless]
        t_eq (float): Time at matter-radiation equality in seconds. # [s]
        
    Returns:
        torch.Tensor: Shape (N, 3) - Accelerations in m/s^2. # [m s^-2]
    """
    
    # 1. Calculate pairwise gravitational acceleration (O(N^2) step)
    # Get all pairwise separation vectors: r_j - r_i
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # Shape (N, N, 3) # [m]
    
    # Get all pairwise distances: |r_j - r_i|
    # Add a small epsilon (softening) to avoid r=0 singularities
    r_norm = torch.norm(diff, dim=2) + 1e-9  # Shape (N, N) # [m]
    
    # Calculate G * m_j / r^3
    # masses.T gives shape (1, N), r_norm gives (N, N)
    factor = G * masses.T / (r_norm**3)  # Shape (N, N) # [m^3 kg^-1 s^-2] * [kg] / [m^3] = [s^-2]
    
    # Multiply factor by separation vectors (diff)
    # (N, N, 1) * (N, N, 3) -> (N, N, 3)
    # This is the acceleration vector on 'i' from 'j'
    accel_grav_matrix = factor.unsqueeze(2) * diff # [s^-2] * [m] = [m s^-2]
    
    # Sum over all 'j' particles to get the total gravitational accel on 'i'
    accel_grav = torch.sum(accel_grav_matrix, dim=1)  # Shape (N, 3) # [m s^-2]
    
    # 2. Calculate Cosmological "Drag" (Expansion) Force
    # In the radiation-dominated era (t < t_eq), a(t) ~ t^(1/2)
    # a_dot = 0.5 * a / t
    # a_ddot = 0.5 * a_dot / t - 0.5 * a / t^2
    #        = 0.5 * (0.5 * a / t) / t - 0.5 * a / t^2
    #        = 0.25 * a / t^2 - 0.5 * a / t^2
    #        = -0.25 * a / t^2
    # So, a_ddot / a = -0.25 / t^2
    
    # This check is crucial. The formula changes in the matter era.
    if t < t_eq:
        a_ddot_over_a = -0.25 / (t**2) # [s^-2]
    else:
        # In matter era (t > t_eq), a(t) ~ t^(2/3)
        # a_dot = (2/3) * a / t
        # a_ddot = (2/3) * a_dot / t - (2/3) * a / t^2
        #        = (2/3) * ((2/3) * a / t) / t - (2/3) * a / t^2
        #        = (4/9) * a / t^2 - (6/9) * a / t^2
        #        = - (2/9) * a / t^2
        # So, a_ddot / a = -2.0 / (9.0 * t**2)
        a_ddot_over_a = -2.0 / (9.0 * t**2) # [s^-2]

    accel_cosmo = a_ddot_over_a * positions  # Shape (N, 3) # [s^-2] * [m] = [m s^-2]
    
    # Total acceleration
    return accel_grav + accel_cosmo # [m s^-2]

def get_acceleration_late_universe(positions, masses, G):
    """
    Calculates the acceleration on N bodies in the LATE universe (inside a halo).
    Here, the system is gravitationally collapsed, so we *only* use Newtonian gravity.
    
    Args:
        positions (torch.Tensor): Shape (N, 3) - Proper positions in meters. # [m]
        masses (torch.Tensor): Shape (N, 1) - Masses in kg. # [kg]
        G (float): Gravitational constant in SI units. # [m^3 kg^-1 s^-2]
        
    Returns:
        torch.Tensor: Shape (N, 3) - Accelerations in m/s^2. # [m s^-2]
    """
    # 1. Calculate pairwise gravitational acceleration (O(N^2) step)
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # Shape (N, N, 3) # [m]
    r_norm = torch.norm(diff, dim=2) + 1e-9  # Shape (N, N) # [m]
    factor = G * masses.T / (r_norm**3)  # Shape (N, N) # [s^-2]
    accel_grav_matrix = factor.unsqueeze(2) * diff # [m s^-2]
    accel_grav = torch.sum(accel_grav_matrix, dim=1)  # Shape (N, 3) # [m s^-2]
    
    return accel_grav

def leapfrog_kick_drift_kick(positions, velocities, masses, accel_func, dt, t, **kwargs):
    """
    Performs one step of the Kick-Drift-Kick (KDK) leapfrog integrator.
    
    Args:
        positions (torch.Tensor): Shape (N, 3) # [m]
        velocities (torch.Tensor): Shape (N, 3) # [m s^-1]
        masses (torch.Tensor): Shape (N, 1) # [kg]
        accel_func (function): The function to call for accelerations 
                               (e.g., get_acceleration_early_universe).
        dt (float): Timestep in seconds. # [s]
        t (float): Current time in seconds. # [s]
        **kwargs: Extra arguments for the accel_func (like G, t_eq).
        
    Returns:
        (torch.Tensor, torch.Tensor): Updated positions and velocities.
    """
    
    # KICK (half step)
    accel_start = accel_func(positions, masses, t, **kwargs) # [m s^-2]
    velocities_half = velocities + accel_start * (dt / 2.0) # [m s^-1]
    
    # DRIFT (full step)
    positions_new = positions + velocities_half * dt # [m]
    
    # KICK (half step)
    # Note: we use t + dt for the acceleration calculation, as required by KDK
    accel_end = accel_func(positions_new, masses, t + dt, **kwargs) # [m s^-2]
    velocities_new = velocities_half + accel_end * (dt / 2.0) # [m s^-1]
    
    return positions_new, velocities_new


