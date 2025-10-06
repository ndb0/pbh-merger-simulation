import torch
import numpy as np
import os
import time
import argparse
import json
from config import load_config
from pbh_population import generate_pbh_population
import cosmo

# --- Constants for Physics Calculations ---
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.98847e30
MPC_M = 3.0857e22
YEAR_S = 3.154e7
GPC3_M3 = (1e9 * MPC_M)**3

def calculate_merger_rate(cfg):
    """
    Calculates the PBH merger rate based on the physics from Raidal et al. (2404.08416).
    This version correctly models the initial angular momentum 'j'.
    """
    # --- 0. Load and Validate Configuration ---
    try:
        f_pbh = cfg.pbh_population.f_pbh
        mu_solar = cfg.pbh_population.f_pbh_mu
        output_dir = cfg.output.save_path
    except AttributeError as e:
        print(f"❌ Configuration Error: Missing key in YAML file: {e}")
        return None

    start_time = time.time()
    print(f"✅ Running calculation for f_pbh = {f_pbh:.1e}")

    # --- 1. Calculate Expected Angular Momentum 'j_tau' ---
    # This is the crucial step derived from the paper (Eq. 39).
    # It calculates the typical angular momentum for a binary of a given mass
    # that is expected to merge at the present time (t_0).
    t0_gyr = 13.8
    M_solar_avg = 2 * mu_solar # For an equal mass binary
    eta_avg = 0.25 # For an equal mass binary

    # Eq. 39 from the paper:
    j_tau = 1.7e-2 * (t0_gyr / 13.8)**(3/37) * (M_solar_avg)**(5/37) * \
            (4 * eta_avg)**(3/37) * f_pbh**(10/37)

    print(f"-> Calculated typical angular momentum for mergers today: j_tau = {j_tau:.2e}")

    # --- 2. Calculate the Unsuppressed Merger Rate (R_E2^(0)) ---
    # This is the raw merger rate before considering disruption effects.
    # It is derived directly from Eq. 42 in the paper.

    # We use Gpc^-3 yr^-1 as the final unit.
    rate_prefactor = 1.6e6 # from Eq. 42

    # Calculate the dependencies on f_pbh, mass (M), and mass ratio (eta)
    fpbh_term = f_pbh**(53/37)
    eta_term = eta_avg**(-34/37)
    M_term = M_solar_avg**(-32/37)
    t_term = (t0_gyr / 13.8)**(-34/37) # This is (t/t_0)^(-34/37), so it's 1.

    # The paper's formula gives the rate density. To get the total rate for
    # a monochromatic mass function, we simplify.
    raw_rate = rate_prefactor * fpbh_term * eta_term * M_term * t_term
    print(f"-> Raw (unsuppressed) merger rate: {raw_rate:.2e} Gpc^-3 yr^-1")

    # --- 3. Apply Suppression Factors from the Paper ---
    sigma_M_sq = 0.005 # from paper footnote 7
    sigma_M = np.sqrt(sigma_M_sq)

    # N_bar_y from Eq. 41: Average number of PBHs in the exclusion radius 'y'
    N_bar_y = (M_solar_avg / mu_solar) * (f_pbh / (f_pbh + sigma_M))

    # S_E suppression factor approximation (Eq. 46 & 47)
    # For a monochromatic mass function, <m^2>/<m>^2 = 1
    m_ratio_term = 1.0

    # The paper uses a complex formula (Eq. 47) for a factor C. For simplicity
    # and stability, we use a well-known approximation which is valid for f_pbh << 1.
    if f_pbh < 0.1:
         S_E = np.exp(-N_bar_y)
    else: # For larger f_pbh, perturbations are more significant
        term1 = m_ratio_term / (N_bar_y + 1e-9)
        term2 = sigma_M_sq / f_pbh**2
        S_E = np.exp(-N_bar_y) * (term1 + term2)**(-21/74)

    # S_L suppression factor approximation (Eq. 60)
    S_L = min(1.0, 0.01 * (f_pbh)**(-0.65) * np.exp(0.03 * np.log(f_pbh)**2)) if f_pbh > 1e-6 else 1.0

    suppressed_rate = raw_rate * S_E * S_L
    print(f"-> Suppression factors: S_E={S_E:.3f}, S_L={S_L:.3f}")
    print(f"-> Suppressed merger rate: {suppressed_rate:.2e} Gpc^-3 yr^-1")

    # --- 4. Save results to a file ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result = {
        'f_pbh': f_pbh,
        'raw_rate_gpc_yr': raw_rate,
        'suppressed_rate_gpc_yr': suppressed_rate,
        'S_E': S_E,
        'S_L': S_L,
    }

    output_filename = os.path.join(output_dir, f'merger_rate_fpbh_{f_pbh:.1e}.json')
    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"\n✅ Results saved to '{output_filename}'")
    print(f"Total calculation time: {time.time() - start_time:.2f} seconds.")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PBH merger rates based on Raidal et al. 2024.")
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    calculate_merger_rate(cfg)

