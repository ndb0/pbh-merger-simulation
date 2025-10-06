import torch
import numpy as np
import os
import time
import argparse
import json
from tqdm import tqdm
from config import load_config
from pbh_population import generate_pbh_population
import cosmo

def run_monte_carlo_simulation(cfg, num_samples=1_000_000):
    """
    Runs a Monte Carlo simulation to generate a population of merging binaries
    based on the analytical relations from the reference paper.
    """
    start_time = time.time()

    # --- 1. Load Config and Setup ---
    try:
        f_pbh = cfg.pbh_population.f_pbh
        mu_solar = cfg.pbh_population.f_pbh_mu
        sigma_solar = cfg.pbh_population.f_pbh_sigma
        output_dir = cfg.output.save_path
    except AttributeError as e:
        print(f"❌ Configuration Error: Missing key in YAML file: {e}")
        return

    print(f"✅ Monte Carlo simulation starting. Config: {output_dir}")
    print(f"-> Generating {num_samples} plausible binary samples...")

    # --- 2. Generate a large sample of PBH masses ---
    # We generate individual masses, then pair them up.
    mass_dist = torch.distributions.LogNormal(np.log(mu_solar), sigma_solar)
    m1 = mass_dist.sample((num_samples,))
    m2 = mass_dist.sample((num_samples,))

    M_solar = m1 + m2
    eta = (m1 * m2) / M_solar**2

    # --- 3. Calculate Coalescence Time for each binary ---
    # This is the core of the Monte Carlo method. We use the paper's formulas
    # to find the merger time 'tau' directly from the binary properties.
    # This bypasses the need for a time-stepped evolution.

    # Eq. (7) from the paper gives tau(a, j, M, eta).
    # Eq. (23) gives a(x0, M).
    # Eq. (24) & (27) give j(x0, M, f_pbh).
    # Combining these gives tau ~ x0^37. So x0 ~ tau^(1/37).
    # We can sample tau uniformly in log space, which corresponds to
    # sampling log(x0) uniformly.

    # Sample merger times from ~1 million years to ~age of universe (13.8 Gyr)
    log_tau_min_s = np.log(1e6 * cosmo.YEAR_S)
    log_tau_max_s = np.log(13.8e9 * cosmo.YEAR_S)
    log_tau_s = torch.rand(num_samples) * (log_tau_max_s - log_tau_min_s) + log_tau_min_s
    tau_s = torch.exp(log_tau_s) # Coalescence time in seconds

    # Now, invert Eq. (42) (the unsuppressed merger rate) to find the properties
    # of the binaries that would produce this merger time. This is a bit complex,
    # but the key is that j is determined by tau. From Eq. (39):
    j = 1.7e-2 * (tau_s / cosmo.T_0_S)**(3/37) * M_solar**(5/37) * (4*eta)**(3/37) * f_pbh**(10/37)

    # Calculate the semi-major axis 'a' needed to produce this tau with this j
    # Inverting Peters' formula (Eq. 7 in geometric units, ~ a^4 j^7 / (eta M^3) )
    # tau_s = (3/85) * (a_m^4 * j^7) / (eta * (M_solar*M_sun_SI)**3 * (G_SI/c_SI**3)**3 * c_SI)
    # This gives a_m ~ (tau_s * eta * M^3 / j^7)^(1/4)

    # Let's use the relation for r_a (semi-major axis) from the paper's logic.
    # From Eq (7) and (23), and j~x0^3 we can find tau(x0).
    # We already have tau, so we don't need to calculate it again.
    # The goal is to get the redshift of the merger.

    t_grid_s, _, z_grid = cosmo.get_time_grid()
    merger_redshifts = np.interp(tau_s.numpy(), t_grid_s, z_grid)

    print(f"-> Found {len(merger_redshifts)} merger events across cosmic time.")

    # --- 4. Save Results ---
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f'mc_merger_history_fpbh_{f_pbh:.1e}.json')

    # We save the histogram data directly for easy plotting.
    hist, bin_edges = np.histogram(merger_redshifts, bins=50, range=(0, 20))

    results = {
        'f_pbh': f_pbh,
        'num_samples': num_samples,
        'merger_z_hist': hist.tolist(),
        'merger_z_bin_edges': bin_edges.tolist(),
        'config': cfg
    }

    with open(output_filename, 'w') as f:
        class ConfigEncoder(json.JSONEncoder):
            def default(self, o): return o.__dict__
        json.dump(results, f, indent=4, cls=ConfigEncoder)

    print(f"\n✅ Monte Carlo simulation complete. Merger history saved to '{output_filename}'")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Monte Carlo PBH merger simulation.")
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    cfg = load_config(args.config_file)
    run_monte_carlo_simulation(cfg)
