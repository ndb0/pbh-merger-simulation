import torch
import cosmo

def get_acceleration_early_universe(positions, masses, t, G, a_eq, t_eq):
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    r_norm = torch.norm(diff, dim=2) + 1e-9
    factor = G * masses.T / (r_norm**3)
    accel_grav_matrix = factor.unsqueeze(2) * diff
    accel_grav = torch.sum(accel_grav_matrix, dim=1)

    if t < t_eq:
        a_ddot_over_a = -0.25 / (t**2)
    else:
        a_ddot_over_a = -2.0 / (9.0 * t**2)

    accel_cosmo = a_ddot_over_a * positions
    return accel_grav + accel_cosmo

def get_acceleration_late_universe(positions, masses, G):
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    r_norm = torch.norm(diff, dim=2) + 1e-9
    factor = G * masses.T / (r_norm**3)
    accel_grav_matrix = factor.unsqueeze(2) * diff
    accel_grav = torch.sum(accel_grav_matrix, dim=1)
    return accel_grav

def leapfrog_kick_drift_kick(positions, velocities, masses, accel_func, dt, t, **kwargs):
    accel_start = accel_func(positions, masses, t, **kwargs)
    velocities_half = velocities + accel_start * (dt / 2.0)
    positions_new = positions + velocities_half * dt
    accel_end = accel_func(positions_new, masses, t + dt, **kwargs)
    velocities_new = velocities_half + accel_end * (dt / 2.0)
    return positions_new, velocities_new
