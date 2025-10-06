import torch
import cosmo  # Your cosmology constants file

# --- SI Constants for high-precision calculations ---
M_sun_SI = 1.98847e30  # Unit: kg | Dimension: M
G_SI = 6.67430e-11     # Unit: m^3 kg^-1 s^-2 | Dimension: L^3 M^-1 T^-2
c_SI = 299792458.0     # Unit: m s^-1 | Dimension: L T^-1

# --- Astronomical Units for N-body simulation ---
# Using this G keeps units consistent for the N-body part without large/small numbers.
# It avoids mixing km/s with seconds directly in the Verlet integrator.
G_astro = 4.30091e-9  # Unit: Mpc (km/s)^2 / M_sun

def compute_accelerations(masses, positions):
    """
    Calculates Newtonian accelerations for an N-body system.
    This function operates in a consistent set of astronomical units.
    """
    # --- Input Units ---
    # masses: M_sun
    # positions: Mpc

    N = masses.shape[0]
    device = positions.device

    # --- Calculation Steps ---
    # r_vec: Difference in positions | Unit: Mpc | Dimension: L
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    r_vec = diff + torch.eye(N, device=device).unsqueeze(2) * 1e-9  # Mask self-interaction

    # r_sq: Squared distance | Unit: Mpc^2 | Dimension: L^2
    r_sq = torch.sum(r_vec**2, dim=2) + 1e-12  # Epsilon for stability
    # r: Distance | Unit: Mpc | Dimension: L
    r = torch.sqrt(r_sq)

    # m_j: Mass of interacting bodies | Unit: M_sun | Dimension: M
    m_j = masses.view(1, N, 1)

    # --- Force and Acceleration Calculation ---
    # Equation: a = G * m / r^2
    # Unit check: (Mpc (km/s)^2 / M_sun) * M_sun / Mpc^2 = (km/s)^2 / Mpc
    # The result is acceleration in units of (km/s)^2 / Mpc, which is dimensionally L T^-2.
    # This is unusual but consistent for the integrator if dt is in units of Mpc/(km/s).
    # We will use a more standard Verlet integrator with dt in seconds later.
    # To get a standard acceleration (e.g., Mpc/s^2), we'd need to convert G.
    # Let's keep it simple for now and convert units in the evolution step.

    # Corrected acceleration formula for standard units. Let's use G in Mpc^3 M_sun^-1 s^-2
    # G_Mpc_s = G_SI * (M_sun_SI / cosmo.Mpc**3) -> ~4.5e-38 (very small)
    # Sticking with G_astro is fine as long as we are consistent.

    # Let's adjust for standard velocity units (km/s) and time (s).
    # a = F/m = G * m_j / r^2
    # The force vector direction is r_vec / r
    acc_vector = -G_astro * m_j * r_vec / r.unsqueeze(2)**3

    # acc: Total acceleration on each body | Unit: (km/s)^2 / Mpc
    acc = torch.sum(acc_vector, dim=1)  # Sum over all other masses j
    return acc


def update_angular_momentum_j(j, a, e, m1, m2, dt):
    """
    Updates orbital parameters due to Gravitational Wave emission using Peters' equations.
    This function converts inputs to SI units for the physics calculation.
    """
    # --- Input Units & Dimensions ---
    # j: dimensionless | Dimension: [1]
    # a: Mpc | Dimension: L
    # e: dimensionless | Dimension: [1]
    # m1, m2: M_sun | Dimension: M
    # dt: seconds | Dimension: T

    device = j.device
    dtype = torch.float64  # Use high precision for stability

    # Promote all inputs to float64 for the calculation
    j, a, e, m1, m2 = j.to(dtype), a.to(dtype), e.to(dtype), m1.to(dtype), m2.to(dtype)
    dt = torch.tensor(dt, device=device, dtype=dtype)

    # --- Unit Conversion to SI ---
    # a_m: semi-major axis | Unit: m | Dimension: L
    a_m = a * cosmo.Mpc
    # m1_kg, m2_kg: masses | Unit: kg | Dimension: M
    m1_kg = m1 * M_sun_SI
    m2_kg = m2 * M_sun_SI

    # --- Peters' Equations Coefficient (beta) ---
    # Equation: beta = (64/5) * G^3 * m1 * m2 * (m1+m2) / c^5
    # Unit check: (m^3 kg^-1 s^-2)^3 * kg * kg * kg / (m s^-1)^5
    #           = (m^9 kg^-3 s^-6) * kg^3 / (m^5 s^-5)
    #           = m^4 s^-1
    # beta | Unit: m^4 s^-1 | Dimension: L^4 T^-1
    beta = (64.0 / 5.0) * (G_SI**3 * m1_kg * m2_kg * (m1_kg + m2_kg)) / c_SI**5

    # --- Intermediate dimensionless terms ---
    e2 = e**2
    one_minus_e2 = torch.clamp(1.0 - e2, min=1e-12)

    # --- Rate of change of semi-major axis (da/dt) ---
    # Equation: da/dt = -beta * f(e) / a^3
    # Unit check: (m^4 s^-1) / m^3 = m s^-1
    # da_dt | Unit: m/s | Dimension: L T^-1 (Correct)
    da_dt = -beta / (a_m**3) * (1 + (73/24)*e2 + (37/96)*e2*e2) / (one_minus_e2**(3.5))

    # --- Rate of change of eccentricity (de/dt) ---
    # Equation: de/dt = -beta * g(e) / a^4
    # Unit check: (m^4 s^-1) * 1 / m^4 = s^-1
    # de_dt | Unit: 1/s | Dimension: T^-1 (Correct)
    de_dt = -19 * beta / (12 * a_m**4) * e * (1 + (121/304)*e2) / (one_minus_e2**(2.5))

    # --- Euler Integration Step ---
    # a_new_m: new semi-major axis | Unit: m | Dimension: L
    a_new_m = a_m + da_dt * dt
    # e_new: new eccentricity | Unit: dimensionless | Dimension: [1]
    e_new = torch.clamp(e + de_dt * dt, 0.0, 0.9999)

    # Clamp to prevent numerical instability
    a_new_m = torch.clamp(a_new_m, min=1e-20)

    # --- Final Parameter Calculation ---
    # j_new: new angular momentum | Unit: dimensionless | Dimension: [1]
    j_new = torch.sqrt(torch.clamp(1.0 - e_new**2, min=0.0))

    # --- Convert back to original units ---
    # a_new_mpc: new semi-major axis | Unit: Mpc | Dimension: L
    a_new_mpc = a_new_m / cosmo.Mpc

    # Return all tensors in the original precision
    return j_new.to(j.dtype), a_new_mpc.to(j.dtype), e_new.to(j.dtype)

