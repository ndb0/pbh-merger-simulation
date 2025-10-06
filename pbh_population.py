import torch
import math
import cosmo

def generate_initial_velocities(masses, positions):
    """
    Calculates initial velocities for a virialized N-body system.
    This implementation uses the total potential energy of the system to ensure
    the virial theorem (2*<T> = -<V>) is satisfied for the system as a whole.
    """
    N = masses.shape[0]
    device = masses.device

    if N < 2:
        return torch.zeros_like(positions)

    G_astro = 4.30091e-9
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    r_ij = torch.norm(diff, dim=2) + 1e-12
    m_i = masses.unsqueeze(1)
    m_j = masses.unsqueeze(0)
    potential_energy_matrix = -G_astro * m_i * m_j / r_ij
    total_potential_energy = torch.sum(torch.triu(potential_energy_matrix, diagonal=1))
    total_kinetic_energy = -0.5 * total_potential_energy

    if total_kinetic_energy <= 0:
        return torch.zeros_like(positions)

    total_mass = torch.sum(masses)
    ke_per_particle = total_kinetic_energy * (masses / total_mass)
    v_mag_sq = 2 * ke_per_particle / masses
    v_mag_kms = torch.sqrt(torch.clamp(v_mag_sq, min=1e-9))
    random_dirs = torch.randn((N, 3), device=device)
    random_dirs = random_dirs / torch.norm(random_dirs, dim=1, keepdim=True)
    velocities_kms = v_mag_kms.unsqueeze(1) * random_dirs
    km_to_Mpc = 1.0 / (cosmo.Mpc / 1000.0)
    velocities_Mpc_s = velocities_kms * km_to_Mpc

    return velocities_Mpc_s

def generate_pbh_population(
    N: int,
    mu: float,
    sigma: float,
    volume: float,
    seed: int,
    device: str,
    virialize: bool = True
):
    """
    Generates the initial masses, positions, and velocities for the PBH population.
    If virialize is False, velocities will be zero.
    """
    torch.manual_seed(seed)
    N = int(N)
    log_mu = math.log(mu)
    normal_dist = torch.distributions.Normal(log_mu, sigma)
    masses = torch.exp(normal_dist.sample((N,))).to(device)
    edge_len = volume ** (1./3.)
    positions = (torch.rand((N, 3), device=device) - 0.5) * edge_len

    if virialize:
        velocities = generate_initial_velocities(masses, positions)
    else:
        velocities = torch.zeros_like(positions)

    return masses, positions, velocities

